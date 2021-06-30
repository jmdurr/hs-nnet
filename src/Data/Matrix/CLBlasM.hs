{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Data.Matrix.CLBlasM where

import Control.DeepSeq
import Control.Monad.State.Strict
import Data.BlasM
import qualified Data.ByteString.Char8 as BC
import Data.List (sortOn)
import Data.Map.Strict as M
import Data.Matrix.OpenCL
import Data.Proxy
import qualified Data.Vector.Storable as V
import Data.Word (Word64)
import Debug.Trace
import Foreign.C.Types
import Foreign.Storable
import GHC.Generics
import GHC.TypeLits
import Language.C.Quote.OpenCL
import qualified Language.C.Syntax
import Text.PrettyPrint.Mainland (pretty)
import Text.PrettyPrint.Mainland.Class (ppr)
import Text.Printf
import Data.Time.Clock
import Data.Time.Format

data CLBlasMx a = CLBlasMx
  { clMem :: CLMem a,
    clMxWidth :: Word64,
    clMxHeight :: Word64,
    clMxDepth :: Word64
  }
  deriving (Generic)

instance Show (CLBlasMx a) where
  show (CLBlasMx (CLMem len _ _) w h d) = "len:" <> show len <> " mxsz:" <> show (w, h, d)

instance NFData a => NFData (CLBlasMx a)

data CLBlasState a = CLBlasState
  { clContext :: CLContext,
    clCommandQueue :: CLCommandQueue,
    clDevices :: [CLDeviceID],
    -- , clMemoryPool :: ResourcePool CLMem
    clCompiledFunctions :: Map String (CLProgram, CLKernel),
    clCachedConvolveLayers :: Map (Word64, Word64, Word64) (CLMem CInt),
    clCachedConvolveDyLayers :: Map (Word64, Word64, Word64) (CLMem CInt),
    clCachedConvolveLayerLayers :: Map (Word64, Word64, Word64) (CLMem CInt),
    clCompiledConvolve :: Maybe (CLProgram, CLKernel),
    clCompiledConvolveLayers :: Maybe (CLProgram, CLKernel),
    clCompiledFlip :: Maybe (CLProgram, CLKernel),
    clCompiledSum :: Maybe (CLProgram, CLKernel),
    clCompiledAddLayer :: Maybe (CLProgram, CLKernel),
    clCompiledMultLayer :: Maybe (CLProgram, CLKernel),
    clCompiledSumFlatten :: Maybe (CLProgram, CLKernel),
    clCompiledAdd :: Maybe (CLProgram, CLKernel),
    clCompiledOuter :: Maybe (CLProgram, CLKernel),
    clCompiledSub :: Maybe (CLProgram, CLKernel),
    clCompiledDense :: Map String (CLProgram, CLKernel),
    clCompiledDenseT1 :: Maybe (CLProgram, CLKernel),
    clCompiledDenseT2 :: Maybe (CLProgram, CLKernel),
    clProxy :: Proxy a,
    clMatMult :: Maybe (CLProgram, CLKernel)
  }
  deriving (Generic)

instance (NFData a) => NFData (CLBlasState a)

class (Storable a, Floating a) => CLBlasType a where
  clblasTypeNearZero :: Proxy a -> Double
  clblasTypeCStringRep :: Proxy a -> String

instance CLBlasType CDouble where
  clblasTypeNearZero _ = 1e-12
  clblasTypeCStringRep _ = "double"

instance CLBlasType CFloat where
  clblasTypeNearZero _ = 1e-6
  clblasTypeCStringRep _ = "float"

withCLGpu :: (MonadFail m, MonadIO m) => Proxy a -> StateT (CLBlasState a) m b -> m b
withCLGpu pxy st = do
  ps <- clGetPlatforms
  ds <- clGetDevices (head ps)
  ctx <- clCreateContext ds
  q <- clCreateCommandQueue ctx (head ds) []
  fst <$> runStateT st (CLBlasState ctx q ds M.empty M.empty M.empty M.empty Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing M.empty Nothing Nothing pxy Nothing)

prepCLGpu :: (MonadFail m, MonadIO m) => Proxy a -> m (CLBlasState a)
prepCLGpu pxy = do
  ps <- clGetPlatforms
  ds <- clGetDevices (head ps)
  ctx <- clCreateContext ds
  q <- clCreateCommandQueue ctx (head ds) []

  pure (CLBlasState ctx q ds M.empty M.empty M.empty M.empty Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing M.empty Nothing Nothing pxy Nothing)

execClGpu :: (MonadFail m, MonadIO m) => CLBlasState a -> StateT (CLBlasState a) m b -> m (b, CLBlasState a)
execClGpu st fun = runStateT fun st

cleanClGpu :: (MonadFail m, MonadIO m) => m ()
cleanClGpu = pure ()

numWorkChunks :: Int -> Int -> Int
numWorkChunks chunkSize workSize =
  workSize `div` chunkSize + (if workSize `mod` chunkSize > 0 then 1 else 0)

type CLBlasM m a = StateT (CLBlasState a) m
instance (MonadIO m, CLBlasType a, MonadFail m, Floating a, Storable a, Real a) => BlasM (StateT (CLBlasState a) m) (CLBlasMx a) where
  nearZero = pure $ clblasTypeNearZero (Proxy :: Proxy a)


  -- TODO dry this up with transposed versions
  dense (Matrix (CLBlasMx mem1 w h d)) (Matrix (CLBlasMx mem2 w1 _ _)) = do
    -- liftIO $ printf "dense\n"
    cls <- get
    let ts = denseTileSize (fromIntegral h) (fromIntegral w) (fromIntegral w1)
    let fn = denseCFunctionName ts
    -- liftIO $ printf "add\n"
    kern <- case M.lookup fn (clCompiledDense cls) of
      Just (_, k) -> pure k
      Nothing -> do
        let code = BC.pack $ pretty 120 (ppr $ denseC cls fn ts)
        (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code fn
        put cls {clCompiledDense = M.insert fn (prog, k') (clCompiledDense cls)}
        pure k'
    let items = denseCWorkgroups ts (fromIntegral h) (fromIntegral w1)
    -- tilesize is 16, 8 rows * 16 columns
    memo <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ h * w1 * d) (clProxy cls)
    -- mxo@(Matrix (CLBlasMx memo _ _ _)) <- konst 0.0
    -- liftIO $ putStrLn "run dense kern"
    e <-
      clRunKernel
        (clCommandQueue cls)
        kern
        [CLAPlain (fromIntegral h :: CInt), CLAPlain (fromIntegral w :: CInt), CLAPlain (fromIntegral w1 :: CInt), CLAPlain (fromIntegral d :: CInt), CLAPlain (0 :: CInt), CLAMem mem1, CLAMem mem2, CLAMem memo]
        items
    -- liftIO $ putStrLn "collect dense kern"
    clWaitForEvent e
    -- liftIO $ putStrLn "end dense kern"

    pure (Matrix (CLBlasMx memo w1 h d))

  denseT1 (Matrix (CLBlasMx mem1 w1 h2 d)) (Matrix (CLBlasMx mem2 w2 _ _)) = do
        -- liftIO $ printf "dense\n"
    cls <- get
    let ts = denseTileSize (fromIntegral w1) (fromIntegral h2) (fromIntegral w2)
    let fn = denseCFunctionName ts
    -- liftIO $ printf "add\n"
    kern <- case M.lookup fn (clCompiledDense cls) of
      Just (_, k) -> pure k
      Nothing -> do
        let code = BC.pack $ pretty 120 (ppr $ denseC cls fn ts)
        (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code fn
        put cls {clCompiledDense = M.insert fn (prog, k') (clCompiledDense cls)}
        pure k'
    let items = denseCWorkgroups ts (fromIntegral w1) (fromIntegral w2)
    -- tilesize is 16, 8 rows * 16 columns
    memo <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ w1 * w2 * d) (clProxy cls)
    -- mxo@(Matrix (CLBlasMx memo _ _ _)) <- konst 0.0
    -- liftIO $ putStrLn ("run denset1 kern" <> show (w1,h2,w2,d))
    -- t <- liftIO $ getCurrentTime
    e <-
      clRunKernel
        (clCommandQueue cls)
        kern
        [CLAPlain (fromIntegral w1 :: CInt), CLAPlain (fromIntegral h2 :: CInt), CLAPlain (fromIntegral w2 :: CInt), CLAPlain (fromIntegral d :: CInt), CLAPlain (1 :: CInt), CLAMem mem1, CLAMem mem2, CLAMem memo]
        items
    -- liftIO $ putStrLn "wait denset1 kern"
    clWaitForEvent e
    -- t' <- liftIO $ getCurrentTime
    -- liftIO $ putStrLn (formatTime defaultTimeLocale "end denset1 kern - %0Es"  (diffUTCTime t' t))
    pure (Matrix (CLBlasMx memo w2 w1 d))

  denseT2 (Matrix (CLBlasMx mem1 w1 h1 d)) (Matrix (CLBlasMx mem2 _ h2 _)) = do
        -- liftIO $ printf "dense\n"
    cls <- get
    let ts = denseTileSize (fromIntegral h1) (fromIntegral w1) (fromIntegral h2)
    let fn = denseCFunctionName ts
    -- liftIO $ printf "add\n"
    kern <- case M.lookup fn (clCompiledDense cls) of
      Just (_, k) -> pure k
      Nothing -> do
        let code = BC.pack $ pretty 120 (ppr $ denseC cls fn ts)
        (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code fn
        put cls {clCompiledDense = M.insert fn (prog, k') (clCompiledDense cls)}
        pure k'
    let items = denseCWorkgroups ts (fromIntegral h1) (fromIntegral h2)
    -- tilesize is 16, 8 rows * 16 columns
    memo <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ h1 * h2 * d) (clProxy cls)
    -- mxo@(Matrix (CLBlasMx memo _ _ _)) <- konst 0.0
    -- liftIO $ putStrLn "run denset2 kern"
    e <-
      clRunKernel
        (clCommandQueue cls)
        kern
        [CLAPlain (fromIntegral h1 :: CInt), CLAPlain (fromIntegral w1 :: CInt), CLAPlain (fromIntegral h2 :: CInt), CLAPlain (fromIntegral d :: CInt), CLAPlain (2 :: CInt), CLAMem mem1, CLAMem mem2, CLAMem memo]
        items
    -- liftIO $ putStrLn "wait denset2 kern"
    clWaitForEvent e
    -- liftIO $ putStrLn "end denset2 kern"
    pure (Matrix (CLBlasMx memo h2 h1 d))


  -- outer, multiply each element of mx1 with each element of mx2
  outer mx1@(Matrix (CLBlasMx mem1 _ l1 _)) mx2@(Matrix (CLBlasMx mem2 _ l2 _)) =
    
    do
      -- liftIO $ printf "add\n"
      cls <- get
      memo <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ l1 * l2) (clProxy cls)
      kern <- case clCompiledOuter cls of
        Just (_, k) -> pure k
        Nothing -> do
          let code = BC.pack $ pretty 120 (ppr $ outerC cls)
          (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code "outerC"
          put cls {clCompiledOuter = Just (prog, k')}
          pure k'
      -- 64 work items in a wavefront
      let numtilesX = l1
      let numtilesY = l2 `div` 128 + if l2 `mod` 128 > 0 then 1 else 0
      e <- clRunKernel (clCommandQueue cls) kern [CLAPlain (fromIntegral l1 :: CInt), CLAPlain (fromIntegral l2 :: CInt), CLAMem mem1, CLAMem mem2,  CLAMem memo] ((fromIntegral (numtilesX), fromIntegral 1), Just (fromIntegral (numtilesY*64), fromIntegral 64), Nothing)
      clWaitForEvent e
      pure (Matrix (CLBlasMx memo l2 l1 1))
    

  scale mx amt = mx `applyFunction` Mul Value (Const amt)

  -- TODO make a generic optC builder, perhaps move all compiled kernels to a map
  add (Matrix (CLBlasMx mem1 w h d)) (Matrix (CLBlasMx mem2 _ _ _)) =
    do
      -- liftIO $ printf "add\n"
      cls <- get
      memo <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ w * h * d) (clProxy cls)
      kern <- case clCompiledAdd cls of
        Just (_, k) -> pure k
        Nothing -> do
          let code = BC.pack $ pretty 120 (ppr $ opC cls "addC" (\l r -> [cexp|$exp:l + $exp:r|]))
          (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code "addC"
          put cls {clCompiledAdd = Just (prog, k')}
          pure k'

      let tilesizeX = min 16 w
      let numtilesX = w `div` 16 + if w `mod` 16 > 0 then 1 else 0
      let tilesizeY = min 16 h
      let numtilesY = h `div` 16 + if h `mod` 16 > 0 then 1 else 0
      e <- clRunKernel (clCommandQueue cls) kern [CLAMem mem1, CLAMem mem2, CLAPlain (fromIntegral w :: CInt), CLAPlain (fromIntegral h :: CInt), CLAMem memo] ((fromIntegral (numtilesX * tilesizeX), fromIntegral tilesizeX), Just (fromIntegral (numtilesY * tilesizeY), fromIntegral tilesizeY), Just (fromIntegral d, 1))
      clWaitForEvent e
      pure (Matrix (CLBlasMx memo w h d))

  addToAllWithDepth (Matrix (CLBlasMx mem1 w1 h1 d1)) (Matrix (CLBlasMx mem2 _ _ _)) = do
    -- liftIO $ printf "adddepth\n"
    cls <- get
    kern <- case clCompiledAddLayer cls of
      Just (_, k) -> pure k
      Nothing -> do
        let code = BC.pack $ pretty 120 (ppr $ addLayerC cls)
        (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code "addLayers"
        put cls {clCompiledAddLayer = Just (prog, k')}
        pure k'
    memo <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ w1 * h1 * d1) (clProxy cls)
    -- split into 8
    -- local work size is the important bit, not number of workgroups
    -- 16x16 is max local size
    -- tileSizeX = 16, tileSizeY = 16
    let tilesizeX = min 16 w1
    let numtilesX = w1 `div` 16 + if w1 `mod` 16 > 0 then 1 else 0
    let tilesizeY = min 16 h1
    let numtilesY = h1 `div` 16 + if h1 `mod` 16 > 0 then 1 else 0
    e <- clRunKernel (clCommandQueue cls) kern [CLAMem mem1, CLAMem mem2, CLAPlain (fromIntegral w1 :: CInt), CLAPlain (fromIntegral h1 :: CInt), CLAMem memo] ((fromIntegral (numtilesX * tilesizeX), fromIntegral tilesizeX), Just (fromIntegral (numtilesY * tilesizeY), fromIntegral tilesizeY), Just (fromIntegral d1, 1))
    clWaitForEvent e
    pure (Matrix (CLBlasMx memo w1 h1 d1))

  multToAllWithDepth (Matrix (CLBlasMx mem1 w1 h1 d1)) (Matrix (CLBlasMx mem2 _ _ _)) = do
    -- liftIO $ printf "multdepth\n"
    cls <- get
    kern <- case clCompiledMultLayer cls of
      Just (_, k) -> pure k
      Nothing -> do
        let code = BC.pack $ pretty 120 (ppr $ multLayerC cls)
        (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code "multLayers"
        put cls {clCompiledMultLayer = Just (prog, k')}
        pure k'
    let tilesizeX = min 16 w1
    let numtilesX = w1 `div` 16 + if w1 `mod` 16 > 0 then 1 else 0
    let tilesizeY = min 16 h1
    let numtilesY = h1 `div` 16 + if h1 `mod` 16 > 0 then 1 else 0

    memo <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ w1 * h1 * d1) (clProxy cls)
    e <- clRunKernel (clCommandQueue cls) kern [CLAPlain (fromIntegral w1 :: CInt), CLAPlain (fromIntegral h1 :: CInt), CLAMem mem1, CLAMem mem2, CLAMem memo] ((fromIntegral (numtilesX * tilesizeX), fromIntegral tilesizeX), Just (fromIntegral (numtilesY * tilesizeY), fromIntegral tilesizeY), Just (fromIntegral d1, 1))
    clWaitForEvent e
    pure (Matrix (CLBlasMx memo w1 h1 d1))

  subtractM (Matrix (CLBlasMx mem1 w h d)) (Matrix (CLBlasMx mem2 _ _ _)) =
    do
      -- liftIO $ printf "add\n"
      cls <- get
      memo <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ w * h * d) (clProxy cls)
      kern <- case clCompiledSub cls of
        Just (_, k) -> pure k
        Nothing -> do
          let code = BC.pack $ pretty 120 (ppr $ opC cls "subC" (\l r -> [cexp|$exp:l - $exp:r|]))
          (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code "subC"
          put cls {clCompiledSub = Just (prog, k')}
          pure k'

      let tilesizeX = min 16 w
      let numtilesX = w `div` 16 + if w `mod` 16 > 0 then 1 else 0
      let tilesizeY = min 16 h
      let numtilesY = h `div` 16 + if h `mod` 16 > 0 then 1 else 0
      e <- clRunKernel (clCommandQueue cls) kern [CLAMem mem1, CLAMem mem2, CLAPlain (fromIntegral w :: CInt), CLAPlain (fromIntegral h :: CInt), CLAMem memo] ((fromIntegral (numtilesX * tilesizeX), fromIntegral tilesizeX), Just (fromIntegral (numtilesY * tilesizeY), fromIntegral tilesizeY), Just (fromIntegral d, 1))
      clWaitForEvent e
      pure (Matrix (CLBlasMx memo w h d))

  applyFunction (Matrix (CLBlasMx mem1 w1 h1 d1)) func =
    let hsh = hashFuncToStr func
        fname = "clblasm_" <> hsh
        sz = w1 * h1 * d1
        code cls = BC.pack $ pretty 120 (ppr $ applyC cls (Language.C.Syntax.Id fname) func)
        tilesizeY = min 16 h1
        numtilesY = h1 `div` tilesizeY + if h1 `mod` tilesizeY > 0 then 1 else 0
        tilesizeX = min 16 w1
        numtilesX = w1 `div` tilesizeX + if w1 `mod` tilesizeX > 0 then 1 else 0
     in do
          -- liftIO $ printf "applyf\n"
          cls <- get
          mem2 <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral sz) (clProxy cls)
          k <- case M.lookup hsh (clCompiledFunctions cls) of
            Nothing -> do
              (prog, kern) <- clKernelFromSource (clContext cls) (clDevices cls) (code cls) fname
              put cls {clCompiledFunctions = M.insert hsh (prog, kern) (clCompiledFunctions cls)}
              pure kern
            Just (_, kern) -> pure kern
          cs <- constPtr func
          -- tile height
          e <- clRunKernel (clCommandQueue cls) k [CLAPlain (fromIntegral w1 :: CInt), CLAPlain (fromIntegral h1 :: CInt), CLAMem cs, CLAMem mem1, CLAMem mem2] ((fromIntegral (numtilesX * tilesizeX), fromIntegral tilesizeX), Just (fromIntegral (numtilesY * tilesizeY), fromIntegral tilesizeY), Just (fromIntegral d1, 1))
          clWaitForEvent e
          pure (Matrix (CLBlasMx mem2 w1 h1 d1))

  konst v = konstH v

  reshapeM = pure . reshapeMh

  mxFromVec = mxFromVecH

  mult (Matrix (CLBlasMx mx w h d)) (Matrix (CLBlasMx mx2 _ _ _)) =
    do
      -- liftIO $ printf "mult\n"
      cls <- get
      kern <- case clMatMult cls of
        Just (_, k) -> pure k
        Nothing -> do
          let code = BC.pack $ pretty 120 (ppr $ multC cls)
          (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code "multC"
          put cls {clMatMult = Just (prog, k')}
          pure k'
      mem3 <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ w * h * d) (clProxy cls)
      let tilesizeX = min 16 w
      let numtilesX = w `div` 16 + if w `mod` 16 > 0 then 1 else 0
      let tilesizeY = min 16 h
      let numtilesY = h `div` 16 + if h `mod` 16 > 0 then 1 else 0
      e <- clRunKernel (clCommandQueue cls) kern [CLAMem mx, CLAMem mx2, CLAPlain (fromIntegral w :: CInt), CLAPlain (fromIntegral h :: CInt), CLAMem mem3] ((fromIntegral (numtilesX * tilesizeX), fromIntegral tilesizeX), Just (fromIntegral (numtilesY * tilesizeY), fromIntegral tilesizeY), Just (fromIntegral d, 1))
      clWaitForEvent e
      pure (Matrix (CLBlasMx mem3 w h d))

  convolveLayers inmx@(Matrix (CLBlasMx _ iw ih ind)) fmx@(Matrix (CLBlasMx _ dyw dyh dyd)) stride pad dialateIn dialateF =
    let layermap = sortOn (\(_, _, c) -> c) [(i, f, f * ind + i) | f <- [0 .. (dyd -1)], i <- [0 .. (ind -1)]]
        -- id,fd,od
        fltmap = concatMap (\(a, b, c) -> [fromIntegral a, fromIntegral b, fromIntegral c]) layermap :: [CInt]
     in do
          -- liftIO $ printf "convolveLayers %d %d\n" (ind* iw * ih) (dyd*dyw*dyh)
          cls <- get
          m <- case M.lookup (ind, dyd, ind * dyd) (clCachedConvolveLayerLayers cls) of
            Nothing -> do
              m' <- clBufferFromList (clContext cls) (clCommandQueue cls) fltmap
              put (cls {clCachedConvolveLayerLayers = M.insert (ind, dyd, ind * dyd) m' (clCachedConvolveLayerLayers cls)})
              pure m'
            Just m' -> pure m'

          memo <- konst 0.0
          es <- convolveLayerStack inmx fmx memo stride pad dialateIn dialateF m (fromIntegral ind)
          clWaitForEvent es
          -- liftIO $ printf "done conl\n"
          pure memo

  convolve inmx@(Matrix (CLBlasMx _ iw ih ind)) fmx@(Matrix (CLBlasMx _ fw fh fd)) _ stride padding dialateIn dialateF =
    let layers = sortOn (\(_, _, c) -> c) [(i, o * ind + i, o) | o <- [0 .. (fd `div` ind - 1)], i <- [0 .. (ind -1)]]
        layermap = concatMap (\(i, f, o) -> [fromIntegral i, fromIntegral f, fromIntegral o]) layers
     in do
          -- liftIO $ printf "convolve %d %d\n" (ind * iw * ih) (fd * fw * fh)
          cls <- get
          m <- case M.lookup (ind, fd, fd `div` ind) (clCachedConvolveLayers cls) of
            Nothing -> do
              m' <- clBufferFromList (clContext cls) (clCommandQueue cls) layermap
              put (cls {clCachedConvolveLayers = M.insert (ind, fd, fd `div` ind) m' (clCachedConvolveLayers cls)})
              pure m'
            Just m' -> pure m'

          -- liftIO $ printf "convolve\n"
          --
          memo <- konst 0.0
          e <- convolveLayerStack inmx fmx memo stride padding dialateIn dialateF m (fromIntegral ind)
          clWaitForEvent e
          pure memo

  {-
     o1 = i1*f1 + i2*f2 + i3*f3
     o2 = i1*f4 + i2*f5 + i3*f6

     given outputs and filters what are input depths  input,filter,output (2,256,128)
     i1 = o1f1 +o2f4  (0,0,0) (1,3,0)
     i2 = o1f2 + o2f5 (0,1,1) (1,4,1)
     i3 = o1f3 + o2f6 (0,2,2) (1,5,2)

     ind = 2
     fd = 256
     od = 128
  -}

  convolveLayersDy inmx@(Matrix (CLBlasMx memi iw ih ind)) fmx@(Matrix (CLBlasMx memf fw fh fd)) stride pad dialateIn dialateF =
    do
      -- liftIO $ printf "convolveDy %d %d\n" (iw * ih * ind) (fw * fh * fd)
      cls <- get
      m <- case M.lookup (ind, fd, od) (clCachedConvolveDyLayers cls) of
        Nothing -> do
          m'@(CLMem len _ _) <- clBufferFromList (clContext cls) (clCommandQueue cls) fltmap
          when (len /= fromIntegral (length fltmap)) (fail "memory length does not match filtermap length")
          put (cls {clCachedConvolveDyLayers = M.insert (ind, fd, od) m' (clCachedConvolveDyLayers cls)})
          pure m'
        Just m' -> pure m'
      memo <- konst 0.0
      -- liftIO $ putStrLn $ show fltmap -- input = dy, filter = filter, output = input
      e <- convolveLayerStack inmx fmx memo stride pad dialateIn dialateF m (fromIntegral ind)
      clWaitForEvent e
      pure memo
    where
      od = fd `div` ind -- 128
      idxs = sortOn (\(_, _, c) -> c) [(o, o * od + i, i) | o <- [0 .. (ind - 1)], i <- [0 .. (od - 1)]]
      -- TODO sort by depth so that we can just through input depth for each filter instead of looping through all
      fltmap = concatMap (\(i, f, o) -> [fromIntegral i, fromIntegral f, fromIntegral o]) idxs :: [CInt]

  sumLayers = sumLayersH

  mxToVec (Matrix (CLBlasMx mem1 w h d)) = do
    -- liftIO $ printf "mxv\n"

    cls <- get
    --      1,2,1
    r <- V.take (fromIntegral $ w * h * d) . V.map (fromRational . toRational) <$> clEnqueueReadVec (clCommandQueue cls) mem1
    -- h = rows, w = cols
    when (V.length r < fromIntegral (w * h * d)) (fail "memory length does not match matrix size")
    pure r
  flip (Matrix (CLBlasMx mem w1 h1 d1)) axis = do
    -- liftIO $ printf "flip\n
    cls <- get

    kern <- case clCompiledFlip cls of
      Just (_, k) -> pure k
      Nothing -> do
        let code = BC.pack $ pretty 120 (ppr $ flipC cls)
        (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code "flipMx"
        put cls {clCompiledFlip = Just (prog, k')}
        pure k'

    let tilesizeY = min 16 h1
    let numtilesY = h1 `div` tilesizeY + if h1 `mod` tilesizeY > 0 then 1 else 0
    let tilesizeX = min 16 w1
    let numtilesX = w1 `div` tilesizeX + if w1 `mod` tilesizeX > 0 then 1 else 0
    memo <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ w1 * h1 * d1) (clProxy cls)
    e <-
      clRunKernel
        (clCommandQueue cls)
        kern
        [ CLAPlain (fromIntegral w1 :: CInt),
          CLAPlain (fromIntegral h1 :: CInt),
          CLAPlain (if axis == FlipX || axis == FlipBoth then 1 else 0 :: CInt),
          CLAPlain (if axis == FlipY || axis == FlipBoth then 1 else 0 :: CInt),
          CLAMem mem,
          CLAMem memo
        ]
        ((fromIntegral (numtilesX * tilesizeX), fromIntegral tilesizeX), Just (fromIntegral (numtilesY * tilesizeY), fromIntegral tilesizeY), Just (fromIntegral d1, 1))
    clWaitForEvent e
    pure (Matrix (CLBlasMx memo w1 h1 d1))

mxFromVecH ds pw ph pd =
  let w' = fromIntegral $ natVal pw
      h' = fromIntegral $ natVal ph
      d' = fromIntegral $ natVal pd
   in do
        cls <- get
        mem <- clBufferFromVec (clContext cls) (clCommandQueue cls) (V.map (fromRational . toRational) ds)
        pure (Matrix (CLBlasMx mem w' h' d'))

-- switch these to pattern match on proxy instead
ctypes :: CLBlasType a => CLBlasState a -> Language.C.Syntax.Type
ctypes cls =
  if clblasTypeCStringRep (clProxy cls) == "double"
    then [cty|double|]
    else [cty|float|]

ctypes4 :: CLBlasType a => CLBlasState a -> Language.C.Syntax.Type
ctypes4 cls =
  if clblasTypeCStringRep (clProxy cls) == "double"
    then [cty|double4|]
    else [cty|float4|]

ctypes8 :: CLBlasType a => CLBlasState a -> Language.C.Syntax.Type
ctypes8 cls =
  if clblasTypeCStringRep (clProxy cls) == "double"
    then [cty|double8|]
    else [cty|float8|]

reshapeMh ::
  forall w h d w2 h2 d2 a.
  (KnownNat w, KnownNat h, KnownNat d, KnownNat w2, KnownNat h2, KnownNat d2, KnownNat (w GHC.TypeLits.* h), KnownNat (w2 GHC.TypeLits.* h2), (w GHC.TypeLits.* h GHC.TypeLits.* d) ~ (w2 GHC.TypeLits.* h2 GHC.TypeLits.* d2)) =>
  Matrix (CLBlasMx a) w h d ->
  Matrix (CLBlasMx a) w2 h2 d2
reshapeMh (Matrix (CLBlasMx mem _ _ _)) =
  let w' = fromIntegral $ natVal (Proxy :: Proxy w2)
      h' = fromIntegral $ natVal (Proxy :: Proxy h2)
      d' = fromIntegral $ natVal (Proxy :: Proxy d2)
   in mkmx w' h' d'
  where
    mkmx :: Word64 -> Word64 -> Word64 -> Matrix (CLBlasMx a) w2 h2 d2
    mkmx nw nh nd = Matrix (CLBlasMx mem nw nh nd)

konstH ::
  forall w h d a m.
  (KnownNat w, KnownNat h, KnownNat d, MonadState (CLBlasState a) m, MonadIO m, MonadFail m, BlasM m (CLBlasMx a), Storable a, Floating a) =>
  Double ->
  m (Matrix (CLBlasMx a) w h d)
konstH v =
  let w = fromIntegral $ natVal (Proxy :: Proxy w)
      h = fromIntegral $ natVal (Proxy :: Proxy h)
      d = fromIntegral $ natVal (Proxy :: Proxy d)
   in do
        cls <- get
        m <- clCreateFilledBuffer (clContext cls) (clCommandQueue cls) clMemReadWrite (w * h * d) (fromRational (toRational v) :: a)
        pure (Matrix (CLBlasMx m w h d))

--mxFromVec (V.replicate (w * h * d) (fromRational $ toRational v)) (Proxy :: Proxy w) (Proxy :: Proxy h) (Proxy :: Proxy d)

constPtr :: (Floating a, Storable a, MonadFail m, MonadIO m, MonadState (CLBlasState a) m) => MxFunction -> m (CLMem a)
constPtr mxf = do
  cls <- get
  clBufferFromList (clContext cls) (clCommandQueue cls) (dbs ++ [0.0])
  where
    dbs = go mxf
    go (Const d) = [fromRational (toRational d)]
    go (Exp f1 f2) = go f1 ++ go f2
    go (Log f1) = go f1
    go (Ln f1) = go f1
    go (Div f1 f2) = go f1 ++ go f2
    go (Mul f1 f2) = go f1 ++ go f2
    go (Abs f1) = go f1
    go (Add f1 f2) = go f1 ++ go f2
    go (Sub f1 f2) = go f1 ++ go f2
    go (Min f1 f2) = go f1 ++ go f2
    go (Max f1 f2) = go f1 ++ go f2
    go (Sinh f) = go f
    go (Cosh f) = go f
    go (Sqrt f) = go f
    go (Tanh f) = go f
    go (If (IfGt l r) f1 f2) = go l ++ go r ++ go f1 ++ go f2
    go (If (IfLt l r) f1 f2) = go l ++ go r ++ go f1 ++ go f2
    go (If (IfEq l r) f1 f2) = go l ++ go r ++ go f1 ++ go f2
    go (If (IfNe l r) f1 f2) = go l ++ go r ++ go f1 ++ go f2
    go Value = []

hashFuncToStr :: MxFunction -> String
hashFuncToStr = go
  where
    go (Const _) = "C"
    go (Exp f1 f2) = "E" ++ go f1 ++ go f2
    go (Log f1) = "L" ++ go f1
    go (Ln f1) = "l" ++ go f1
    go (Div f1 f2) = "d" ++ go f1 ++ go f2
    go (Mul f1 f2) = "m" ++ go f1 ++ go f2
    go (Abs f1) = "a" ++ go f1
    go (Add f1 f2) = "p" ++ go f1 ++ go f2
    go (Sub f1 f2) = "M" ++ go f1 ++ go f2
    go (Min f1 f2) = "x" ++ go f1 ++ go f2
    go (Max f1 f2) = "X" ++ go f1 ++ go f2
    go (Sinh f) = "s" ++ go f
    go (Cosh f) = "c" ++ go f
    go (Sqrt f) = "r" ++ go f
    go (Tanh f) = "t" ++ go f
    go (If (IfGt l r) f1 f2) = "G" ++ go l ++ go r ++ go f1 ++ go f2
    go (If (IfLt l r) f1 f2) = "g" ++ go l ++ go r ++ go f1 ++ go f2
    go (If (IfEq l r) f1 f2) = "e" ++ go l ++ go r ++ go f1 ++ go f2
    go (If (IfNe l r) f1 f2) = "z" ++ go l ++ go r ++ go f1 ++ go f2
    go Value = "v"

withFuncToCode :: Int -> Language.C.Syntax.Type -> Language.C.Syntax.Exp -> MxFunction -> (Language.C.Syntax.Exp -> Language.C.Syntax.Exp) -> (Int, Language.C.Syntax.Exp)
withFuncToCode i t vs fx sf =
  let (i', s') = funcToCode i t vs fx
   in (i', sf s')

seqWithFuncToCode :: Int -> Language.C.Syntax.Type -> Language.C.Syntax.Exp -> MxFunction -> MxFunction -> (Language.C.Syntax.Exp -> Language.C.Syntax.Exp -> Language.C.Syntax.Exp) -> (Int, Language.C.Syntax.Exp)
seqWithFuncToCode i t vs f1 f2 sf =
  let (i1, s1) = funcToCode i t vs f1
      (i2, s2) = funcToCode i1 t vs f2
   in (i2, sf s1 s2)

seqWithFuncToCode3 :: Int -> Language.C.Syntax.Type -> Language.C.Syntax.Exp -> MxFunction -> MxFunction -> MxFunction -> (Language.C.Syntax.Exp -> Language.C.Syntax.Exp -> Language.C.Syntax.Exp -> Language.C.Syntax.Exp) -> (Int, Language.C.Syntax.Exp)
seqWithFuncToCode3 i t vs f1 f2 f3 sf =
  let (i1, s1) = funcToCode i t vs f1
      (i2, s2) = funcToCode i1 t vs f2
      (i3, s3) = funcToCode i2 t vs f3
   in (i3, sf s1 s2 s3)

funcToCode :: Int -> Language.C.Syntax.Type -> Language.C.Syntax.Exp -> MxFunction -> (Int, Language.C.Syntax.Exp)
funcToCode i _ vs Value = (i, vs)
funcToCode i t vs (Exp f1 f2) = seqWithFuncToCode i t vs f1 f2 (\tv vsv -> [cexp|pow($exp:tv,$exp:vsv)|])
funcToCode i t vs (Log f1) = withFuncToCode i t vs f1 (\tv -> [cexp|log10($exp:tv)|])
funcToCode i t vs (Ln f1) = withFuncToCode i t vs f1 (\tv -> [cexp|log($exp:tv)|])
funcToCode i t vs (Div f1 f2) = seqWithFuncToCode i t vs f1 f2 (\tv vsv -> [cexp|(($exp:tv)/($exp:vsv))|])
funcToCode i t vs (Mul f1 f2) = seqWithFuncToCode i t vs f1 f2 (\tv vsv -> [cexp|(($exp:tv)*($exp:vsv))|])
funcToCode i t vs (Abs f1) = withFuncToCode i t vs f1 (\tv -> [cexp|fabs($exp:tv)|])
funcToCode i t vs (Add f1 f2) = seqWithFuncToCode i t vs f1 f2 (\tv vsv -> [cexp|(($exp:tv)+($exp:vsv))|])
funcToCode i t vs (Sub f1 f2) = seqWithFuncToCode i t vs f1 f2 (\tv vsv -> [cexp|(($exp:tv)-($exp:vsv))|])
funcToCode i _ _ (Const _) = (i + 1, [cexp|consts[ $int:i ]|])
funcToCode i t vs (Min f1 f2) = seqWithFuncToCode i t vs f1 f2 (\tv vsv -> [cexp|fmin($exp:tv,$exp:vsv)|])
funcToCode i t vs (Max f1 f2) = seqWithFuncToCode i t vs f1 f2 (\tv vsv -> [cexp|fmax($exp:tv,$exp:vsv)|])
funcToCode i t vs (Sinh f) = withFuncToCode i t vs f (\tv -> [cexp|sinh($exp:tv)|])
funcToCode i t vs (Cosh f) = withFuncToCode i t vs f (\tv -> [cexp|cosh($exp:tv)|])
funcToCode i t vs (Sqrt f) = withFuncToCode i t vs f (\tv -> [cexp|sqrt($exp:tv)|]) --
funcToCode i t vs (Tanh f) = withFuncToCode i t vs f (\tv -> [cexp|tanh($exp:tv)|])
funcToCode i t vs (If expr f1 f2) =
  let (i', s) = ifExpToCode i expr t vs
   in seqWithFuncToCode i' t vs f1 f2 (\e1 e2 -> [cexp|(($exp:s)?($exp:e1):($exp:e2))|])

ifExpToCode :: Int -> IfExp -> Language.C.Syntax.Type -> Language.C.Syntax.Exp -> (Int, Language.C.Syntax.Exp)
ifExpToCode i (IfGt l r) t vs = seqWithFuncToCode i t vs l r (\e1 e2 -> [cexp|($exp:e1)>($exp:e2)|])
ifExpToCode i (IfLt l r) t vs = seqWithFuncToCode i t vs l r (\e1 e2 -> [cexp|($exp:e1)<($exp:e2)|])
ifExpToCode i (IfEq l r) t vs = seqWithFuncToCode i t vs l r (\e1 e2 -> [cexp|($exp:e1)==($exp:e2)|])
ifExpToCode i (IfNe l r) t vs = seqWithFuncToCode i t vs l r (\e1 e2 -> [cexp|($exp:e1)!=($exp:e2)|])

sumLayersH :: forall w h d a m. (CLBlasType a, MonadState (CLBlasState a) m, MonadIO m, MonadFail m, BlasM m (CLBlasMx a), Storable a, Floating a, KnownNat w, KnownNat h, KnownNat d) => Matrix (CLBlasMx a) w h d -> m (Matrix (CLBlasMx a) 1 1 d)
sumLayersH (Matrix (CLBlasMx mem1 wm hm dpt)) = do
  cls <- get
  kern <- case clCompiledSum cls of
    Just (_, k) -> pure k
    Nothing -> do
      let code = BC.pack $ pretty 120 (ppr $ sumC cls)
      (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code "sumC"
      put cls {clCompiledSum = Just (prog, k')}
      pure k'
  -- this should result in one number per depth
  memo <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral dpt) (clProxy cls)
  e <-
    clRunKernel
      (clCommandQueue cls)
      kern
      [ CLAPlain (fromIntegral wm :: CInt),
        CLAPlain (fromIntegral hm :: CInt),
        CLAMem mem1,
        CLAMem memo
      ]
      ((fromIntegral dpt, 1), Nothing, Nothing)
  clWaitForEvent e
  pure (Matrix (CLBlasMx memo 1 1 dpt) :: Matrix (CLBlasMx a) 1 1 d)

-- | called with x,y tiles and processes full depth
addLayerC :: CLBlasType a => CLBlasState a -> Language.C.Syntax.Func
addLayerC cls =
  let typ = ctypes cls
   in [cfun|kernel void addLayers(const __constant $ty:typ * mx, __constant $ty:typ* v, int w, int h,  global $ty:typ * out){
          int x = get_global_id(0);
          int y = get_global_id(1);
          int d = get_global_id(2);
          if (x >= w || y >= h) return;
          int off = d * w * h + y*w + x;
          out[off] = mx[off] + v[d];
       } 
     |]

-- must be called with 1 per depth
multLayerC :: CLBlasType a => CLBlasState a -> Language.C.Syntax.Func
multLayerC cls =
  let typ = ctypes cls
   in [cfun|kernel void multLayers(int w, int h, const __constant $ty:typ * mx, const __constant $ty:typ * v, global $ty:typ * out){
           int x = get_global_id(0);
           int y = get_global_id(1);
           int d = get_global_id(2);
           if (x >= w || y >= h) return;
           int off = d * w * h + y * w + x;
           out[off] = mx[off] * v[d];
          }
     |]

multC :: CLBlasType a => CLBlasState a -> Language.C.Syntax.Func
multC cls =
  let typ = ctypes cls
   in [cfun|kernel void multC(const global $ty:typ * mx1, const global $ty:typ * mx2, int w, int h, global $ty:typ * out){
           int x = get_global_id(0);
           int y = get_global_id(1);
           int d = get_global_id(2);
           if (x >= w || y >= h) return;
           int off = d * w * h + y * w + x;
           out[off] = mx1[off] * mx2[off];
          } 
     |]

opC :: CLBlasType a => CLBlasState a -> String -> (Language.C.Syntax.Exp -> Language.C.Syntax.Exp -> Language.C.Syntax.Exp) -> Language.C.Syntax.Func
opC cls fname op =
  let typ = ctypes cls
      opsxp = op [cexp|mx1[off]|] [cexp|mx2[off]|]
   in [cfun|kernel void $id:fname(const global $ty:typ * mx1, const global $ty:typ * mx2, int w, int h, global $ty:typ * out){
           int x = get_global_id(0);
           int y = get_global_id(1);
           int d = get_global_id(2);
           if (x >= w || y >= h) return;
           int off = d * w * h + y * w + x;
           out[off] = $exp:opsxp;
          } 
     |]

flipC :: CLBlasType a => CLBlasState a -> Language.C.Syntax.Func
flipC cls =
  let typ = ctypes cls
   in [cfun|kernel void flipMx(int w, int h, int flipx, int flipy, __constant const $ty:typ * input, global $ty:typ * output){
          int x = get_global_id(0);
          int y = get_global_id(1);
          int d = get_global_id(2);
          if (x >= w || y >= h) return;
          int off = d * w * h + y * w + x;
          int foff = d*w*h + (flipy ? (h-y-1) : y) * w + (flipx ? (w - x - 1) : x);
          output[off] = input[foff];
        }
      |]

-- add up each row into [r x depth] array - allows use of vload
sumC :: CLBlasType a => CLBlasState a -> Language.C.Syntax.Func
sumC cls =
  let typ = ctypes cls
   in [cfun| kernel void sumC(int w, int h, __constant const $ty:typ * input, global $ty:typ * output){
        int d = get_global_id(0);
        int num = w * h;
        int doff = d * num;
        $ty:typ val = 0;
        for (int i = 0; i < num; i++){
          val += input[doff+i];
        }
        output[d] = val;    
      }
    |]

--        ks t = "kernel void " <> fname <> "(const int n, global " <> nm <> "* consts, global " <> nm <> "* A, global " <> nm <> "* B){ int d = get_global_id(0); " <> nm <> " val = A[id]; B[id] = " <> snd (funcToCode 0 t [cexp|val" func) <> ";}"

applyC :: (CLBlasType a1, ToIdent a2) => CLBlasState a1 -> a2 -> MxFunction -> Language.C.Syntax.Func
applyC cls nm func =
  let typ = ctypes cls
      (_, funcexp) = funcToCode 0 typ [cexp|val|] func
   in [cfun|kernel void $id:nm (int w, int h, global $ty:typ * consts, global $ty:typ * input, global $ty:typ * output){
              int x = get_global_id(0);
              int y = get_global_id(1);
              int d = get_global_id(2);
              if (x >= w || y >= h) return;
              int off = d * w * h + y * w + x;
              $ty:typ val = input[off];
              output[off] = $exp:funcexp;
            }
     |]

-- TODO convolve needs to be optimized for smaller kernels ~5x5 < 16x16
{- notes from amd convolution presentation
    * Use 8x8 workgroups
    * Use constant memory for filter with fixed size and float4 data type __constant is different than const
    * Use float4 to process filter and input
    * load input data into local memory
    * pass defines to kernel at runtime

  each workgroup tiles across the output
   - what is local input size?
   - each workitem needs to copy the values that the filter will use directly or workitem[0] can copy the entire local memory
   - given a set of outputs, what inputs are needed?
   - should not be doing a lot of calculation in here
   -
-}
{-
   inputLeft and inputTop are the position in the input with padding and dialation included, so 0,0 usually refers to the top left of the padded area

-}
-- TODO check ranges and also look for values greater than 1000
convolveLayerStack ::
  ( KnownNat w,
    KnownNat h,
    KnownNat d,
    KnownNat fw,
    KnownNat fh,
    KnownNat fd,
    KnownNat ow,
    KnownNat oh,
    KnownNat od,
    KnownNat sx,
    KnownNat sy,
    KnownNat pxl,
    KnownNat pxr,
    KnownNat pyt,
    KnownNat pyb,
    KnownNat dx,
    KnownNat dy,
    KnownNat dfx,
    KnownNat dfy,
    KnownNat (h - 1),
    KnownNat (w - 1),
    (((h + ((h - 1) GHC.TypeLits.* dy)) + pyt + pyb - (fh + ((fh - 1) GHC.TypeLits.* dfy)) + sy) ~ (sy GHC.TypeLits.* oh)),
    (((w + ((w - 1) GHC.TypeLits.* dx)) + pxl + pxr - (fw + ((fw - 1) GHC.TypeLits.* dfx)) + sx) ~ (sx GHC.TypeLits.* ow)),
    Mod ((w + ((w - 1) GHC.TypeLits.* dx)) + pxl + pxr - (fw + ((fw - 1) GHC.TypeLits.* dfx))) sx ~ 0,
    Mod ((h + ((h - 1) GHC.TypeLits.* dy)) + pyt + pyb - (fh + ((fh - 1) GHC.TypeLits.* dfy))) sy ~ 0,
    CLBlasType a,
    MonadIO m,
    MonadFail m
  ) =>
  Matrix (CLBlasMx a) w h d -> -- 6 + 0 + 0 = 5 ~ 2 * 3
  Matrix (CLBlasMx a) fw fh fd ->
  Matrix (CLBlasMx a) ow oh od ->
  -- | stride of filter
  (Proxy sx, Proxy sy) ->
  -- | padding of input each side specified separately for type math
  (Proxy pxl, Proxy pxr, Proxy pyt, Proxy pyb) ->
  -- | dialation of input, 0 is a valid known nat...
  (Proxy dx, Proxy dy) ->
  -- | dialation of the filter
  (Proxy dfx, Proxy dfy) ->
  -- | mapping of layers to convolve (input,filter,output)
  CLMem CInt ->
  CInt ->
  StateT (CLBlasState a) m CLEvent
convolveLayerStack (Matrix (CLBlasMx memi iw ih ind)) (Matrix (CLBlasMx memf fw fh fd)) (Matrix (CLBlasMx memo@(CLMem len _ _) ow oh od)) (strideX, strideY) (padXL, padXR, padYT, padYB) (dialateX, dialateY) (dialateFX, dialateFY) mapping@(CLMem maplen _ _) outsPerMapping =
  let sx = natVal strideX
      sy = natVal strideY
      pxl = natVal padXL
      pyt = natVal padYT
      pxr = natVal padXR
      pyb = natVal padYB
      dx = natVal dialateX
      dy = natVal dialateY
      dfx = natVal dialateFX
      dfy = natVal dialateFY
   in do
        cls <- get
        kern <- case clCompiledConvolveLayers cls of
          Just (_, k) -> pure k
          Nothing -> do
            let code = BC.pack $ pretty 120 (ppr $ convolveC cls)
            (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code "convolve"
            put cls {clCompiledConvolveLayers = Just (prog, k')}
            pure k'
        -- try this without using local memory, tile size is 1 per output
        let baseTileSizeX = min 16 ow -- number of outputs per workgroup, 1 worker per work item
        let baseTileSizeY = min 16 oh

        let tileNumX = ow `div` baseTileSizeX + if ow `mod` baseTileSizeX > 0 then 1 else 0
        let tileNumY = oh `div` baseTileSizeY + if oh `mod` baseTileSizeY > 0 then 1 else 0
        -- this must be run with each output depth handled in 1 workitem

        -- liftIO $ printf "convolve bt (%d,%d)  s (%d,%d) f (%d,%d) df (%d,%d) ts (%d,%d)  tn (%d,%d) d (%d)\n" baseTileSizeX baseTileSizeY sx sy fw fh dfx dfy tileSizeX tileSizeY tileNumX tileNumY outDepth
        clRunKernel
          (clCommandQueue cls)
          kern
          [ CLAMem memi,
            CLAMem memf,
            CLAMem memo,
            CLAMem mapping,
            CLAPlain (fromIntegral maplen :: CInt),
            CLAPlain (fromIntegral iw :: CInt),
            CLAPlain (fromIntegral ih :: CInt),
            CLAPlain (fromIntegral outsPerMapping :: CInt),
            CLAPlain (fromIntegral fw :: CInt),
            CLAPlain (fromIntegral fh :: CInt),
            CLAPlain (fromIntegral ow :: CInt),
            CLAPlain (fromIntegral oh :: CInt),
            CLAPlain (fromIntegral sx :: CInt),
            CLAPlain (fromIntegral sy :: CInt),
            CLAPlain (fromIntegral pxl :: CInt),
            CLAPlain (fromIntegral pxr :: CInt),
            CLAPlain (fromIntegral pyt :: CInt),
            CLAPlain (fromIntegral pyb :: CInt),
            CLAPlain (fromIntegral dx :: CInt),
            CLAPlain (fromIntegral dy :: CInt),
            CLAPlain (fromIntegral dfx :: CInt),
            CLAPlain (fromIntegral dfy :: CInt)
          ]
          ((fromIntegral (tileNumX * baseTileSizeX), baseTileSizeX), Just (fromIntegral (tileNumY * baseTileSizeY), baseTileSizeY), Just (fromIntegral od, 1))

convolveC :: CLBlasType a => CLBlasState a -> Language.C.Syntax.Func
convolveC cls =
  let typ = ctypes cls
   in [cfun|kernel void convolve(__constant const $ty:typ *input,
                      const __constant $ty:typ* filter,
                      global $ty:typ* output,
                      __constant int* mapping, int mappingSize,
                      int inputW, int inputH, int mappingsPerDepth,
                      int fw, int fh,
                      int ow, int oh,
                      int strideX, int strideY, int padXL, int padXR, int padYT, int padYB,
                      int dialateX, int dialateY,
                      int dialateFX, int dialateFY
                      ) {
          int outX = get_global_id(0);
          int outY = get_global_id(1);
          int outputD = get_global_id(2);
          if (outX >= ow) return;
          if (outY >= oh) return;
          int oOff = outputD * ow * oh;
          $ty:typ result = 0;
          int preInX = outX * strideX - padXL; // position in possibly dialated matrix
          int mpStart = outputD * mappingsPerDepth * 3;
          for (int mp = mpStart; mp < mpStart + (mappingsPerDepth * 3); mp += 3){
             if (mapping[mp+2] != outputD) {
               printf("mapping depth does not match output depth %d %d",mapping[mp+2],outputD);
               } // each output depth fully handled by 1 work group
             int inputD = mapping[mp];
             int filterD = mapping[mp+1];
             int inDOff = inputD * inputW * inputH;
             int fdOff = filterD * fw * fh;
             for (int j = 0; j < fh; j++){
                int inj = j * (dialateFY + 1);
                int fjOff = fdOff + j * fw;      
                int inY = outY * strideY + inj - padYT;
                
                
                if (inY >= 0 && inY % (dialateY+1) == 0){
                    inY = inY / (dialateY + 1);
                    int injOff = inDOff + inY * inputW;
                    for (int i = 0; i < fw; i++){
                        int ini = i * (dialateFX+1);
                        int inX = preInX + ini; // position of filter cell in input and dialation
                        if (inX >= 0 && inX % (dialateX + 1) == 0){
                           inX = inX / (dialateX + 1);
                           if (inX < inputW && inY < inputH){
                              $ty:typ tmp = input[injOff + inX] * filter[fjOff+i];
                              result += tmp;
                              //if (tmp > 1000 || tmp < -1000){
                              // printf("f: (%d,%d,%d,%d) in: (%d:%d,%d:%d,%d,%d) vals: (%.2f,%.2f)",i,j,fjOff+i,filterD,inX,inputW,inY,inputH,injOff,inputD,input[injOff+inX],filter[fjOff+i]);
                             //}
                           } 
                        }
                    }
                }
              }
           }

              //if (result > 1000 || result < -1000) {
              //   printf("number unusually large ooff %d oy %d ox %d result %.2f od %d ow %d oh %d",oOff,outY,outX,result,outputD,ow,oh);
              //}
           
                output[oOff+outY*ow+outX] += result;
              
          }
      |]

-- 32 threads
-- 64k local memory
-- 128 total registers (use 96 for rows)
-- 256 work group size
-- 1024x1024x1024 work item size
-- pref workgroup size 32
-- wavefront width 32

-- 16,1,16
-- 1024x1024 -> 240ms
-- 1024x1 * 1x1024 -> 1.2
-- 1x1024 * 1024x1024 -> 823us

-- numthreads = arows * bColTileSize / colsPerThread
--     arows,bcols,acoltilesize,colsPerThread
-- t64   8,8,4     - 9.3ms  700us  240us
-- t256 16,16,1    - 250ms  2.5ms  3.1ms
-- t64   8,8,1     - 135ms  710us  1.25ms
-- t64   8,16,1,2  - 113ms  656us  1.28ms
-- t16   4,16,1,4  - 50.6ms 179us  977us


-- t32  
{-
    wavefront is 32
    workgroup size 32
    64k local mem
    128 registers per

    32 * 4 numbers = 256 computations

    32 threads per workgroup
     at least 8 computations per thread + * 4 numbers
    16x16 tile of B = 256 numbers


   1024x1024 * 1024x1024
   TS 8x8x4  11.65ms
   TS 16x16x4 5.53ms

   1x1024 * 1024x1024
   TS 1x64x2  353ms
   TS 8x8x2   335ms
   TS 8x8x4   356ms
   TS 1x64x2  355ms
   TS 1x64x1x16 362ms

   1024x1 * 1x1024
   TS 32x32x1 3.51s
   TS 32x32x4 3.67s
   TS 16x16x1 384ms
   TS 8x16x1  346ms
   TS 8x8x1   339ms
   TS 8x8x1x16 342ms
   TS 32x8x1x32 354ms
   TS 8x8x4x32  167us


-}
 
data TileSize = TileSize {rowsLeft :: Int, colsRight :: Int, colsLeft :: Int, tileThreads :: Int}

-- tilethreads must be divisible by rowsLeft
-- colsLeft * rowsLeft must be divisible by tileThreads
-- rowsLeft * colsRight must be divisible by tileThreads
denseTileSize :: Int -> Int -> Int -> TileSize
denseTileSize m n k
    | m == 1 && k == 1 = TileSize 16 16 1 32
    | m == 1 && n > 2 = TileSize 1 4 4 4
    | m == 1 && n > 1 = TileSize 1 4 4 4
    | m == 1 = TileSize 1 4 4 4
    | n > 3 = TileSize 16 16 4 64 --4 64 = 4ms, 8 128 - 6ms
    | n > 1 = TileSize 16 16 2 32
    | otherwise = TileSize 8 8 4 32 
  -- | m < 32 && n < 32 && k < 32 = TileSize 8 8 4
  -- | m == 1 = TileSize 1 32 4


denseCWorkgroups :: TileSize -> Int -> Int -> ((Word64, Word64), Maybe (Word64, Word64), Maybe (Word64, Word64))
denseCWorkgroups tilesize m k =
  let roughATiles = m `div` rowsLeft tilesize
      roughBTiles = k `div` colsRight tilesize
      numATiles = if roughATiles * rowsLeft tilesize < m then roughATiles + 1 else roughATiles
      numBTiles = if roughBTiles * colsRight tilesize < k then roughBTiles + 1 else roughBTiles
   in -- total work items = m * k
     -- traceShowId
        ( (fromIntegral (numATiles * rowsLeft tilesize), fromIntegral (rowsLeft tilesize)),
          Just (fromIntegral (numBTiles * (tileThreads tilesize `div` rowsLeft tilesize)), fromIntegral (tileThreads tilesize `div` rowsLeft tilesize)),
          Nothing
        )
denseCFunctionName :: TileSize -> String
denseCFunctionName tilesize =
  printf "dense_%d_%d_%d_%d" (rowsLeft tilesize) (colsRight tilesize) (colsLeft tilesize) (tileThreads tilesize)

denseC :: CLBlasType a => CLBlasState a -> String -> TileSize -> Language.C.Syntax.Func
denseC cls fname tilesize =
  let typ = ctypes cls
      numThreads = tileThreads tilesize
      bColsPerTile = colsRight tilesize
      aColsPerTile = colsLeft tilesize
      aRowsPerTile = rowsLeft tilesize
      outputsPerThread = (aRowsPerTile * bColsPerTile) `div` numThreads
      sharedRows = aRowsPerTile < numThreads
      localSize
        | sharedRows = aRowsPerTile * aColsPerTile
        | otherwise = aRowsPerTile * aColsPerTile
      -- with transpose, 1 = transpose left matrix
      --                 2 = transpose right matrix
      -- M,N,K should represent the sizes after any transpose
   in [cfun|kernel void $id:fname(int M, int N, int K, int D, int T,
                      const global $ty:typ * mx1,
                      const global $ty:typ * mx2,
                      global $ty:typ* output
                      ){
                        const size_t threadId = get_local_id(0) * get_local_size(1) + get_local_id(1);
//                        printf("tid: %d\n",threadId);
                        local $ty:typ localBCols[$int:bColsPerTile*$int:aColsPerTile];
                        local $ty:typ localARows[$int:localSize];
                        for (int d = 0; d < D; d++){
                          const int startRow = (threadId * $int:outputsPerThread) / $int:bColsPerTile;
                          const int startCol = (threadId * $int:outputsPerThread) % $int:bColsPerTile;

                          $ty:typ result[$int:outputsPerThread];
                          for (int o = 0; o < $int:outputsPerThread; o++){
                            result[o] = 0.0;
                          }
                          const int aRowOff = get_group_id(0) * $int:aRowsPerTile;
                          const int bColOff = get_group_id(1) * $int:bColsPerTile;

                          for (int acol = 0; acol < N; acol += $int:aColsPerTile){
                            // read A Rows
                            int toRead = max($int:localSize / $int:numThreads,1); //8
                            for (int i = 0; i < toRead && (threadId * toRead + i) < $int:localSize; i++){
                              // a row = aRowOff + ((threadId * toRead + i)/aColsPerTile) 
                              // a col = acol + ((threadId * toRead + i) % aColsPerTile)
                         
                              if (T == 1){
  /*
                              printf("assign d %d A[%d]=%f\n"
                                ,d
                                ,threadId * toRead + i
                                ,mx1[(d*M*N + aRowOff + ((threadId * toRead + i) / $int:aColsPerTile)) + (((threadId * toRead + i) % $int:aColsPerTile) + acol)*M]
                              );
  */

                                // to get to a row I have to skip all depths
                                  localARows[threadId * toRead + i] = 
                                    (aRowOff + (threadId * toRead + i) / $int:aColsPerTile < M && ((threadId * toRead + i) % $int:aColsPerTile + acol < N)) ? 
                                     mx1[(d*M*N + aRowOff + ((threadId * toRead + i) / $int:aColsPerTile)) + (((threadId * toRead + i) % $int:aColsPerTile) + acol)*M] : 
                                     0;
                                
                              } else{  
                                localARows[threadId * toRead + i] = 
                                  (aRowOff + ((threadId * toRead + i) / $int:aColsPerTile) < M && (((threadId * toRead + i) % $int:aColsPerTile) + acol < N)) ? 
                                  mx1[(aRowOff + d * M + ((threadId * toRead + i) / $int:aColsPerTile))*N + ((threadId * toRead + i) % $int:aColsPerTile) + acol] :
                                  0;
                              }
                            }

                            // read B Cols
                            toRead = max(($int:bColsPerTile * $int:aColsPerTile) / $int:numThreads,1);
                            for (int i = 0; i < toRead && (threadId * toRead + i) < ($int:bColsPerTile*$int:aColsPerTile); i++){
                              if (T == 2){
  /*
                                 printf("assign d %d B[%d]=%f\n"
                                  ,d
                                  ,threadId * toRead + i
                                  ,mx2[(((threadId * toRead + i) % $int:aColsPerTile + acol + d * N * K))+((bColOff + (threadId * toRead + i) / $int:aColsPerTile)*N)]
                                );
  */
                                localBCols[threadId * toRead + i] = 
                                  (((threadId * toRead + i) % $int:aColsPerTile + acol) < N && ((bColOff + (threadId * toRead + i) / $int:aColsPerTile) < K)) ?
                                    mx2[(((threadId * toRead + i) % $int:aColsPerTile + acol + d * N * K))+((bColOff + (threadId * toRead + i) / $int:aColsPerTile)*N)] : 
                                    0;
                              } else {
                                localBCols[threadId * toRead + i] = 
                                  (((threadId * toRead + i) % $int:aColsPerTile + acol) < N && ((bColOff + (threadId * toRead + i) / $int:aColsPerTile) < K)) ?

                                    mx2[(((threadId * toRead + i) % $int:aColsPerTile + acol + d * N)*K)+(bColOff + (threadId * toRead + i) / $int:aColsPerTile)] :
                                    0;

                              }
                            }

                            barrier(CLK_LOCAL_MEM_FENCE);
  /*
                            // print out localARow
                              printf("A[%d] = %f\n"
                                , threadId
                                , localARows[threadId]
                              );
  */                        

                            // multiply tiles
                            // total multiplications
                            // printf("outputsperthread %d tid %d aRowOff %d bColOff %d\n",$int:outputsPerThread,threadId,aRowOff,bColOff);
                            
                            // outRow start = (threadId * outputsPerThread) / $int:bColsPerTile
                            // outRow end = ((threadId+1) * outputsPerThread) / $int:bColsPerTile



                            $ty:typ aRow[$int:aColsPerTile];
                            // threads and columns should align on boundaries, so bunch of threads will need to load more data at the same time
                            
                            int currRow = -1;
                            for (int i = 0; i < $int:outputsPerThread; i++){
                              // output is a row, column combo
                              // outRow = (threadId * outputsPerThread + i) / $int:bColsPerTile
                              // outCol = (threadId * outputsPerThread + i) % $int:bColsPerTile
                              // row of A
                              // numCols is calculated per row

                              if (((startCol + i) / $int:bColsPerTile) + startRow + aRowOff < M && (((startCol+i) % $int:bColsPerTile) + bColOff) < K){
                                if (((startCol + i) / $int:bColsPerTile) != currRow){
                                  currRow = (startCol + i) / $int:bColsPerTile;
                                    for (int r = 0; r < $int:aColsPerTile; r++){
  /*
                                      printf("localR (%d,%d) = localARows %d = %f\n"
                                        ,currRow + startRow,r
                                        ,(currRow + startRow) * $int:aColsPerTile + r
                                        ,localARows[(currRow + startRow) * $int:aColsPerTile + r]
                                      );
  */
                                      aRow[r] = localARows[(currRow + startRow) * $int:aColsPerTile + r];
                                    } 
                                }

                                // multiply aRow by column
                                for (int c = 0; c < $int:aColsPerTile && (c+acol) < N; c++){
  /*
                                  printf("result[%d] += aRow[%d]:%f + localBCols(%d,%d):%f = %f"
                                    ,i
                                    ,c
                                    ,aRow[c]
                                    ,((startCol+i) % $int:bColsPerTile)
                                    ,c
                                    ,localBCols[((startCol+i) % $int:bColsPerTile)*$int:aColsPerTile + c]
                                    , result[i] + aRow[c] * localBCols[((startCol+i) % $int:bColsPerTile)*$int:aColsPerTile + c]
                                  
                                  );
  */
                                  result[i] += (aRow[c] * localBCols[((startCol+i) % $int:bColsPerTile)*$int:aColsPerTile + c]);
                                }
                              }
                            }
                            barrier(CLK_LOCAL_MEM_FENCE);
                          }
                          // copy results to output
                          // start row - aRowOff + ((threadId * $int:outputsPerThread + i) / $int:bColsPerTile);
                          // start col - bColOff + ((threadId * $int:outputsPerThread + i) % $int:bColsPerTile);
                          // TODO switch this to use variables startCol startRow etc
                          // should always process in batches of columns
                          for (int i = 0; i < $int:outputsPerThread; i++){
                            if ((aRowOff + startRow + ((startCol + i) / $int:bColsPerTile)) < M && (bColOff + ((startCol + i) % $int:bColsPerTile)) < K){
  /*
                                printf("tid %d:%d:%d output (%d,%d) = %f\n"
                                  , threadId, i, startCol+i
                                  , (aRowOff+startRow+((startCol + i) / $int:bColsPerTile))
                                  , ((startCol + i) % $int:bColsPerTile)
                                  , result[i]
                                );
                                */
                                output[(aRowOff+d*M+startRow+((startCol + i) / $int:bColsPerTile))*K+(bColOff + (startCol + i) % $int:bColsPerTile)] 
                                  = result[i];                            
                            }
                          }
                        }
                      }
      |]


outerC :: CLBlasType a => CLBlasState a -> Language.C.Syntax.Func
outerC cls =
  let typ = ctypes cls
     
   in [cfun|kernel void outerC(int M, int N,
                      __constant const $ty:typ * mx1,
                      const global $ty:typ * mx2,
                      global $ty:typ* output
                      ){

                        if (get_group_id(0) < M){
                          $ty:typ uval = mx1[get_group_id(0)];
                          const int pos = get_group_id(1)*128 + get_local_id(1);
                          if (pos < N)
                            output[get_global_id(0)*N+get_local_id(1)] = uval * mx2[pos];
                          if (pos + 64 < N)
                            output[get_global_id(0)*N+get_local_id(1)+64] = uval * mx2[pos+64];
                        }
                      }

      |]