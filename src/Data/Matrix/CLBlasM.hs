{-# LANGUAGE DataKinds #-}
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

import Control.Monad (when)
import Control.Monad.State.Strict
import Data.BlasM
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BC
import Data.Hashable
import Data.List (intercalate)
import Data.Map.Strict as M
import Data.Matrix.CLBlas
import Data.Matrix.OpenCL
import Data.Proxy
import Data.Text.Encoding (encodeUtf8)
import qualified Data.Text.Lazy as T
import Data.Word (Word64)
import Debug.Trace
import Foreign.C.Types
import Foreign.Storable
import Formatting
import GHC.TypeLits
import Language.C.Quote.OpenCL
import Text.PrettyPrint.Mainland (pretty)
import Text.PrettyPrint.Mainland.Class (ppr)
import Text.Printf

data NProxy (n :: Nat) = NProxy

data CLBlasMx a = CLBlasMx
  { clMem :: CLMem a,
    clMxWidth :: Word64,
    clMxHeight :: Word64,
    clMxDepth :: Word64
  }

instance Show (CLBlasMx a) where
  show (CLBlasMx (CLMem len _ _) w h d) = "len:" <> show len <> " mxsz:" <> show (w, h, d)

data CLBlasState a = CLBlasState
  { clContext :: CLContext,
    clCommandQueue :: CLCommandQueue,
    clDevices :: [CLDeviceID],
    -- , clMemoryPool :: ResourcePool CLMem
    clCompiledFunctions :: Map String (CLProgram, CLKernel),
    clCompiledConvolve :: Maybe (CLProgram, CLKernel),
    clCompiledConvolveLayers :: Maybe (CLProgram, CLKernel),
    clCompiledFlip :: Maybe (CLProgram, CLKernel),
    clCompiledSum :: Maybe (CLProgram, CLKernel),
    clCompiledAddLayer :: Maybe (CLProgram, CLKernel),
    clCompiledMultLayer :: Maybe (CLProgram, CLKernel),
    clCompiledSumFlatten :: Maybe (CLProgram, CLKernel),
    clProxy :: Proxy a,
    clMatMult :: Maybe (CLProgram, CLKernel)
  }

withCLGpu :: (MonadFail m, MonadIO m) => Proxy a -> StateT (CLBlasState a) m b -> m b
withCLGpu pxy st = do
  ps <- clGetPlatforms
  ds <- clGetDevices (head ps)
  ctx <- clCreateContext ds
  q <- clCreateCommandQueue ctx (head ds) []
  r <- liftIO clblasSetup
  --liftIO $ putStrLn "run state"
  v <- fst <$> runStateT st (CLBlasState ctx q ds M.empty Nothing Nothing Nothing Nothing Nothing Nothing Nothing pxy Nothing)
  --liftIO $ putStrLn "end run"
  liftIO clblasTeardown
  pure v

instance (MonadIO m, CLBlasType a, MonadFail m, Floating a, Storable a, Real a) => BlasM (StateT (CLBlasState a) m) (CLBlasMx a) where
  nearZero = pure $ clblasTypeNearZero (Proxy :: Proxy a)

  dot (Matrix (CLBlasMx mem1 w1 h1 d1)) (Matrix (CLBlasMx mem2 w2 h2 d2)) = do
    cls <- get
    mem <- clCreateBuffer (clContext cls) clMemReadWrite 1 (clProxy cls)
    e <- clblasDot (clContext cls) (clCommandQueue cls) mem1 mem2 mem
    clWaitForEvent e
    r <- clEnqueueReadBuffer (clCommandQueue cls) mem
    case r of
      [] -> fail "Could not dot product, result was empty..."
      (x : _) -> pure $ clblasTypeToDouble x

  dense (Matrix (CLBlasMx mem1@(CLMem len _ _) w h d)) (Matrix (CLBlasMx mem2 w1 _ _)) = do
    cls <- get
    -- 1024 cols, 1600 rows, 1 cols, 1600 cols, 1 row
    -- 1600x1024 * 1x1600 = 1600x1600
    -- (h,w) x (w,w1) = (h,w1)
    -- clblast wants matrix a to be lda*(k-1)+n.. where did that come from?
    -- (cols,rows)
    -- (w,h) (w1,w) = (w1,h)
    --liftIO $ putStrLn $ "mult buffers: " <> intercalate ":" (Prelude.map show [len, w, h, d, w1, h, h])
    mem3 <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ h * w1 * d) (clProxy cls)
    -- m - rows a, n - cols b, k - cols a
    es <- mapM (\((offl, offr), offo) -> clblasGemm clblasrowmajor clblasnotrans clblasnotrans h w1 w 1.0 mem1 offl w mem2 offr w1 0.0 mem3 offo w1 (clCommandQueue cls)) offs
    clWaitEvents es
    pure (Matrix (CLBlasMx mem3 w1 h d))
    where
      -- this won't work for depth > 1
      offsL = [w * h * o | o <- [0 .. (d - 1)]]
      offsR = [w1 * w * o | o <- [0 .. (d - 1)]]
      offsO = [w1 * h * o | o <- [0 .. (d - 1)]]
      offs = zip (zip offsL offsR) offsO

  -- (l1,1) outer (1,l2) = (l1,l2)
  -- sgemm: m -> # rows of a after transpose if needed
  --        n -> # of cols of matrix b after transpose if needed
  --        k -> # of cols of matrix a after transpose if needed
  --        LDA -> # of cols for a if row major, otherwise # of rows
  --        LDB -> # of cols for b if row major, otherwise # of rows
  --        LDC -> # of cols for c if row major, otherwise # of rows
  denseT1 (Matrix (CLBlasMx mem1 w1 h2 d1)) (Matrix (CLBlasMx mem2 w2 _ _)) = do
    cls <- get
    --liftIO $ putStrLn $ "t1 mult buffers: " <> (intercalate ":" $ Prelude.map show [w1, h1, d1, w2, h2])
    -- (w1,h2) * (h2,w2) = w1 * w2
    mem3 <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ w1 * w2 * d1) (clProxy cls)
    es <- mapM (\((offl, offr), offo) -> clblasGemm clblasrowmajor clblastrans clblasnotrans w1 w2 h2 1.0 mem1 offl w1 mem2 offr w2 0.0 mem3 offo w2 (clCommandQueue cls)) offs
    clWaitEvents es
    pure (Matrix (CLBlasMx mem3 w2 w1 d1))
    where
      offsL = [w1 * h2 * o | o <- [0 .. (d1 - 1)]]
      offsR = [w2 * h2 * o | o <- [0 .. (d1 - 1)]]
      offsO = [w1 * w2 * o | o <- [0 .. (d1 - 1)]]
      offs = zip (zip offsL offsR) offsO

  denseT2 (Matrix (CLBlasMx mem1 w1 h1 d1)) (Matrix (CLBlasMx mem2 _ h2 _)) = do
    cls <- get
    mem3 <- clCreateBuffer (clContext cls) clMemReadWrite (h1 * h2 * d1) (clProxy cls)
    -- ma (h1,w1)
    -- mb (h2,w1) -> (w1,h2)T
    -- mc (h1,h2)
    es <- mapM (\((offl, offr), offo) -> clblasGemm clblasrowmajor clblasnotrans clblastrans h1 h2 w1 1.0 mem1 offl w1 mem2 offr w1 0.0 mem3 offo h2 (clCommandQueue cls)) offs
    clWaitEvents es
    pure (Matrix (CLBlasMx mem3 h2 h1 d1)) -- (h1,w1) x (w1,h2) -> (h1,h2)
    where
      -- h w1 w
      offsL = [w1 * h1 * o | o <- [0 .. (d1 - 1)]]
      offsR = [w1 * h2 * o | o <- [0 .. (d1 - 1)]]
      offsO = [h1 * h2 * o | o <- [0 .. (d1 - 1)]]
      offs = zip (zip offsL offsR) offsO

  -- does not work with depth at all
  outer (Matrix (CLBlasMx mem1 _ l1 _)) (Matrix (CLBlasMx mem2 _ l2 _)) = do
    cls <- get
    mem3 <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ l1 * l2) (clProxy cls)
    e <- clblasGemm clblasrowmajor clblasnotrans clblastrans (fromIntegral l1) (fromIntegral l2) 1 1.0 mem1 0 1 mem2 0 1 0.0 mem3 0 (fromIntegral l2) (clCommandQueue cls)
    clWaitForEvent e
    pure (Matrix (CLBlasMx mem3 l2 l1 1))

  scale mx amt = mx `applyFunction` Mul Value (Const amt)

  add mx1@(Matrix (CLBlasMx mem1 w1 h1 d1)) mx2@(Matrix (CLBlasMx mem2 _ _ _)) = do
    cls <- get
    --liftIO $ putStrLn $ "Add matrices: " <> show mx1 <> " + " <> show mx2
    memo <- clblasClone (clContext cls) (clCommandQueue cls) mem1
    e <- clblasAxpy (clCommandQueue cls) mem2 1.0 memo
    clWaitForEvent e
    pure (Matrix (CLBlasMx memo w1 h1 d1))

  addToAllWithDepth mx1@(Matrix (CLBlasMx mem1 w1 h1 d1)) mx2@(Matrix (CLBlasMx mem2 _ _ _)) = do
    cls <- get
    kern <- case clCompiledAddLayer cls of
      Just (_, k) -> pure k
      Nothing -> do
        let code = BC.pack $ pretty 120 (ppr $ addLayerC cls)
        (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code "addLayers"
        put cls {clCompiledAddLayer = Just (prog, k')}
        pure k'
    memo <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ w1 * h1 * d1) (clProxy cls)
    e <- clRunKernel (clCommandQueue cls) kern [CLAPlain (fromIntegral d1 :: CInt), CLAMem mem1, CLAMem mem2, CLAMem memo] (fromIntegral w1, Just (fromIntegral h1), Nothing)
    clWaitForEvent e
    pure (Matrix (CLBlasMx memo w1 h1 d1))

  multToAllWithDepth mx1@(Matrix (CLBlasMx mem1 w1 h1 d1)) mx2@(Matrix (CLBlasMx mem2 _ _ _)) = do
    cls <- get
    kern <- case clCompiledMultLayer cls of
      Just (_, k) -> pure k
      Nothing -> do
        let code = BC.pack $ pretty 120 (ppr $ multLayerC cls)
        (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code "multLayers"
        put cls {clCompiledMultLayer = Just (prog, k')}
        pure k'
    memo <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ w1 * h1 * d1) (clProxy cls)
    e <- clRunKernel (clCommandQueue cls) kern [CLAPlain (fromIntegral d1 :: CInt), CLAMem mem1, CLAMem mem2, CLAMem memo] (fromIntegral w1, Just (fromIntegral h1), Nothing)
    clWaitForEvent e
    pure (Matrix (CLBlasMx memo w1 h1 d1))

  subtractM mx1@(Matrix (CLBlasMx mem1 w1 h1 d1)) mx2@(Matrix (CLBlasMx mem2 w2 h2 d2)) = do
    cls <- get
    --liftIO $ putStrLn $ "Sub matrices: " <> show mx1 <> " - " <> show mx2
    memo <- clblasClone (clContext cls) (clCommandQueue cls) mem1
    --liftIO $ putStrLn "blasaxpy"
    e <- clblasAxpy (clCommandQueue cls) mem2 (-1.0) memo
    --liftIO $ putStrLn "wait for e"
    clWaitForEvent e
    pure (Matrix (CLBlasMx memo w1 h1 d1))

  applyFunction (Matrix (CLBlasMx mem1 w1 h1 d1)) func =
    let hsh = hashFuncToStr func
        fname = "clblasm_" <> hsh
        sz = w1 * h1 * d1
        nm = clblasTypeCStringRep (undefined :: a)
        ks t = "kernel void " <> fname <> "(const int n, global " <> nm <> "* consts, global " <> nm <> "* A, global " <> nm <> "* B){ int id = get_global_id(0); if (id < n && id < get_global_size(0)){" <> nm <> " val = A[id]; B[id] = " <> snd (funcToCode 0 t "val" func) <> ";}}"
        kernel t = BC.pack $ ks t
     in do
          cls <- get
          mem2 <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral sz) (clProxy cls)
          k <- case M.lookup hsh (clCompiledFunctions cls) of
            Nothing -> do
              --liftIO $ putStrLn "kernel f s"
              (prog, kern) <- clKernelFromSource (clContext cls) (clDevices cls) (kernel $ clblasTypeCStringRep (asProxyTypeOf 0.0 (clProxy cls))) fname
              put cls {clCompiledFunctions = M.insert hsh (prog, kern) (clCompiledFunctions cls)}
              pure kern
            -- compile program, update state
            Just (_, kern) -> pure kern
          --liftIO $ putStrLn "kernel s a"
          cs <- constPtr func
          e <- clRunKernel (clCommandQueue cls) k [CLAPlain (fromIntegral sz :: CInt), CLAMem cs, CLAMem mem1, CLAMem mem2] (fromIntegral sz, Nothing, Nothing)
          --liftIO $ putStrLn "kernel w"
          clWaitForEvent e
          --liftIO $ putStrLn "kernel e"
          pure (Matrix (CLBlasMx mem2 w1 h1 d1))

  konst v = konstH v

  reshapeM = pure . reshapeMh

  mxFromList = mxFromListH

  mult (Matrix (CLBlasMx mx w h d)) (Matrix (CLBlasMx mx2 _ _ _)) =
    do
      cls <- get
      kern <- case clMatMult cls of
        Just (_, k) -> pure k
        Nothing -> do
          let rtype = clblasTypeCStringRep (undefined :: a)
          let code = BC.pack $ pretty 120 (ppr $ multC cls)
          --liftIO $ putStrLn "kernel f s"
          (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code "multC"
          put cls {clMatMult = Just (prog, k')}
          pure k'
      mem3 <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ w * h * d) (clProxy cls)
      --liftIO $ putStrLn "kernel s a"
      e <- clRunKernel (clCommandQueue cls) kern [CLAMem mx, CLAMem mx2, CLAMem mem3] (fromIntegral w, Just (fromIntegral h), Just (fromIntegral d))
      --liftIO $ putStrLn "kernel w"
      clWaitForEvent e
      --liftIO $ putStrLn "kernel e"
      pure (Matrix (CLBlasMx mem3 w h d))

  convolve (Matrix (CLBlasMx memi iw ih id)) (Matrix (CLBlasMx memf fw fh fd)) _ (strideX, strideY) (padXL, padXR, padYT, padYB) (dialateX, dialateY) (dialateFX, dialateFY) flipF =
    let sx = fromIntegral $ natVal strideX
        sy = fromIntegral $ natVal strideY
        pxl = fromIntegral $ natVal padXL
        pyt = fromIntegral $ natVal padYT
        pxr = fromIntegral $ natVal padXR
        pyb = fromIntegral $ natVal padYB
        dx = fromIntegral $ natVal dialateX
        dy = fromIntegral $ natVal dialateY
        dfx = fromIntegral $ natVal dialateFX
        dfy = fromIntegral $ natVal dialateFY
     in do
          --liftIO $ putStrLn ("convolve [" <> show iw <> "," <> show ih <> "," <> show id <> "] with filter [" <> show fw <> "," <> show fh <> "," <> show fd <> "]")
          cls <- get
          kern <- case clCompiledConvolve cls of
            Just (_, k) -> pure k
            Nothing -> do
              let code = BC.pack $ pretty 120 (ppr $ convolveC cls)
              (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code "convolve"
              put cls {clCompiledConvolve = Just (prog, k')}
              pure k'

          let ow = ((iw + ((iw - 1) * dx) + pxl + pxr - (fw + ((fw - 1) * dfx))) `div` sx) + 1
          let oh = ((ih + ((ih - 1) * dy) + pyt + pyb - (fh + ((fh - 1) * dfy))) `div` sy) + 1
          let od = fd `div` id
          --clDebugBuffer (clCommandQueue cls) memi "convolve input"
          --clDebugBuffer (clCommandQueue cls) memi "convolve filter"
          memo <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ ow * oh * od) (clProxy cls)
          e <-
            clRunKernel
              (clCommandQueue cls)
              kern
              [ CLAPlain (fromIntegral iw :: CInt),
                CLAPlain (fromIntegral ih :: CInt),
                CLAPlain (fromIntegral id :: CInt),
                CLAMem memi,
                CLAMem memf,
                CLAPlain (fromIntegral fw :: CInt),
                CLAPlain (fromIntegral fh :: CInt),
                CLAPlain (fromIntegral sx :: CInt),
                CLAPlain (fromIntegral sy :: CInt),
                CLAPlain (fromIntegral pxl :: CInt),
                CLAPlain (fromIntegral pxr :: CInt),
                CLAPlain (fromIntegral pyt :: CInt),
                CLAPlain (fromIntegral pyb :: CInt),
                CLAPlain (fromIntegral dx :: CInt),
                CLAPlain (fromIntegral dy :: CInt),
                CLAPlain (fromIntegral dfx :: CInt),
                CLAPlain (fromIntegral dfy :: CInt),
                CLAPlain (if flipF then 1 else 0 :: CInt),
                CLAPlain (if flipF then 1 else 0 :: CInt),
                CLAPlain (fromIntegral ow :: CInt),
                CLAPlain (fromIntegral oh :: CInt),
                CLAMem memo
              ]
              (fromIntegral ow, Just (fromIntegral oh), Just (fromIntegral od))
          clWaitForEvent e
          --clDebugBuffer (clCommandQueue cls) memo "convolve output"

          pure (Matrix (CLBlasMx memo ow oh od))

  convolveLayers (Matrix (CLBlasMx memi iw ih id)) (Matrix (CLBlasMx memf fw fh fd)) (strideX, strideY) (padXL, padXR, padYT, padYB) (dialateX, dialateY) (dialateFX, dialateFY) flipF =
    let sx = fromIntegral $ natVal strideX
        sy = fromIntegral $ natVal strideY
        pxl = fromIntegral $ natVal padXL
        pyt = fromIntegral $ natVal padYT
        pxr = fromIntegral $ natVal padXR
        pyb = fromIntegral $ natVal padYB
        dx = fromIntegral $ natVal dialateX
        dy = fromIntegral $ natVal dialateY
        dfx = fromIntegral $ natVal dialateFX
        dfy = fromIntegral $ natVal dialateFY
     in do
          cls <- get
          kern <- case clCompiledConvolveLayers cls of
            Just (_, k) -> pure k
            Nothing -> do
              let code = BC.pack $ pretty 120 (ppr $ convolveLayersC cls)
              (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code "convolveLayers"
              put cls {clCompiledConvolveLayers = Just (prog, k')}
              pure k'

          let ow = ((iw + ((iw - 1) * dx) + pxl + pxr - (fw + ((fw - 1) * dfx))) `div` sx) + 1
          let oh = ((ih + ((ih - 1) * dy) + pyt + pyb - (fh + ((fh - 1) * dfy))) `div` sy) + 1
          let od = fd * id
          memo <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ ow * oh * od) (clProxy cls)
          e <-
            clRunKernel
              (clCommandQueue cls)
              kern
              [ CLAPlain (fromIntegral iw :: CInt),
                CLAPlain (fromIntegral ih :: CInt),
                CLAPlain (fromIntegral id :: CInt),
                CLAMem memi,
                CLAMem memf,
                CLAPlain (fromIntegral fw :: CInt),
                CLAPlain (fromIntegral fh :: CInt),
                CLAPlain (fromIntegral sx :: CInt),
                CLAPlain (fromIntegral sy :: CInt),
                CLAPlain (fromIntegral pxl :: CInt),
                CLAPlain (fromIntegral pxr :: CInt),
                CLAPlain (fromIntegral pyt :: CInt),
                CLAPlain (fromIntegral pyb :: CInt),
                CLAPlain (fromIntegral dx :: CInt),
                CLAPlain (fromIntegral dy :: CInt),
                CLAPlain (fromIntegral dfx :: CInt),
                CLAPlain (fromIntegral dfy :: CInt),
                CLAPlain (if flipF then 1 else 0 :: CInt),
                CLAPlain (if flipF then 1 else 0 :: CInt),
                CLAPlain (fromIntegral ow :: CInt),
                CLAPlain (fromIntegral oh :: CInt),
                CLAPlain (0 :: CInt),
                CLAMem memo
              ]
              (fromIntegral ow, Just (fromIntegral oh), Just (fromIntegral od))
          clWaitForEvent e
          pure (Matrix (CLBlasMx memo ow oh od))

  convolveLayersDy (Matrix (CLBlasMx memi iw ih id)) (Matrix (CLBlasMx memf fw fh fd)) (strideX, strideY) (padXL, padXR, padYT, padYB) (dialateX, dialateY) (dialateFX, dialateFY) flipF =
    let sx = fromIntegral $ natVal strideX
        sy = fromIntegral $ natVal strideY
        pxl = fromIntegral $ natVal padXL
        pyt = fromIntegral $ natVal padYT
        pxr = fromIntegral $ natVal padXR
        pyb = fromIntegral $ natVal padYB
        dx = fromIntegral $ natVal dialateX
        dy = fromIntegral $ natVal dialateY
        dfx = fromIntegral $ natVal dialateFX
        dfy = fromIntegral $ natVal dialateFY
     in do
          cls <- get
          kern <- case clCompiledConvolveLayers cls of
            Just (_, k) -> pure k
            Nothing -> do
              let code = BC.pack $ pretty 120 (ppr $ convolveLayersC cls)
              (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code "convolveLayers"
              put cls {clCompiledConvolveLayers = Just (prog, k')}
              pure k'

          let ow = ((iw + ((iw - 1) * dx) + pxl + pxr - (fw + ((fw - 1) * dfx))) `div` sx) + 1
          let oh = ((ih + ((ih - 1) * dy) + pyt + pyb - (fh + ((fh - 1) * dfy))) `div` sy) + 1
          let od = fd `div` id
          memo <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ ow * oh * od) (clProxy cls)
          e <-
            clRunKernel
              (clCommandQueue cls)
              kern
              [ CLAPlain (fromIntegral iw :: CInt),
                CLAPlain (fromIntegral ih :: CInt),
                CLAPlain (fromIntegral id :: CInt),
                CLAMem memi,
                CLAMem memf,
                CLAPlain (fromIntegral fw :: CInt),
                CLAPlain (fromIntegral fh :: CInt),
                CLAPlain (fromIntegral sx :: CInt),
                CLAPlain (fromIntegral sy :: CInt),
                CLAPlain (fromIntegral pxl :: CInt),
                CLAPlain (fromIntegral pxr :: CInt),
                CLAPlain (fromIntegral pyt :: CInt),
                CLAPlain (fromIntegral pyb :: CInt),
                CLAPlain (fromIntegral dx :: CInt),
                CLAPlain (fromIntegral dy :: CInt),
                CLAPlain (fromIntegral dfx :: CInt),
                CLAPlain (fromIntegral dfy :: CInt),
                CLAPlain (if flipF then 1 else 0 :: CInt),
                CLAPlain (if flipF then 1 else 0 :: CInt),
                CLAPlain (fromIntegral ow :: CInt),
                CLAPlain (fromIntegral oh :: CInt),
                CLAPlain (1 :: CInt),
                CLAMem memo
              ]
              (fromIntegral ow, Just (fromIntegral oh), Just (fromIntegral od))
          clWaitForEvent e
          pure (Matrix (CLBlasMx memo ow oh od))

  sumFlatten (Matrix (CLBlasMx mx1 w1 h1 d1)) pnd =
    let ndv = fromIntegral $ natVal pnd
     in do
          cls <- get
          kern <- case clCompiledSumFlatten cls of
            Just (_, k) -> pure k
            Nothing -> do
              let code = BC.pack $ pretty 120 (ppr $ sumFlattenC cls)
              (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code "sumFlatten"
              put cls {clCompiledSumFlatten = Just (prog, k')}
              pure k'
          memo <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ w1 * h1 * ndv) (clProxy cls)
          e <-
            clRunKernel
              (clCommandQueue cls)
              kern
              [CLAPlain (fromIntegral d1 :: CInt), CLAPlain (fromIntegral ndv :: CInt), CLAMem mx1, CLAMem memo]
              (fromIntegral w1, Just (fromIntegral h1), Nothing)
          clWaitForEvent e
          pure (Matrix (CLBlasMx memo w1 h1 ndv))

  sumLayers = sumLayersH

  mxToLists (Matrix (CLBlasMx mem1 w h d)) = do
    cls <- get
    --      1,2,1
    r <- Prelude.map (fromRational . toRational) <$> clEnqueueReadBuffer (clCommandQueue cls) mem1
    -- h = rows, w = cols
    when (length r /= fromIntegral (w * h * d)) (fail "memory length does not match matrix size")
    let layers = chunk (fromIntegral $ w * h) r
    pure $ Prelude.map (chunk (fromIntegral w)) layers
    where
      chunk n [] = []
      chunk n xs = Prelude.take n xs : chunk n (Prelude.drop n xs)

  flip (Matrix (CLBlasMx imem w h d)) axis = do
    cls <- get
    kern <- case clCompiledFlip cls of
      Just (_, k) -> pure k
      Nothing -> do
        let code = BC.pack $ pretty 120 (ppr $ flipC cls)
        (prog, k') <- clKernelFromSource (clContext cls) (clDevices cls) code "flipMx"
        put cls {clCompiledFlip = Just (prog, k')}
        pure k'
    memo <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral $ w * h * d) (clProxy cls)
    e <-
      clRunKernel
        (clCommandQueue cls)
        kern
        [ CLAPlain (fromIntegral w :: CInt),
          CLAPlain (fromIntegral h :: CInt),
          CLAPlain (fromIntegral d :: CInt),
          CLAPlain (if axis == FlipX || axis == FlipBoth then 1 else 0 :: CInt),
          CLAPlain (if axis == FlipY || axis == FlipBoth then 1 else 0 :: CInt),
          CLAMem imem,
          CLAMem memo
        ]
        (fromIntegral w, Just (fromIntegral h), Nothing)
    clWaitForEvent e
    pure (Matrix (CLBlasMx memo w h d))

mxFromListH ds pw ph pd =
  let w' = fromIntegral $ natVal pw
      h' = fromIntegral $ natVal ph
      d' = fromIntegral $ natVal pd
   in do
        --liftIO $ putStrLn $ "mxFromListH" <> show w' <> ":" <> show h' <> ":" <> show d' <> " - len list " <> show (length ds)
        cls <- get
        mem <- clBufferFromList (clContext cls) (clCommandQueue cls) (Prelude.map (fromRational . toRational) ds)
        --liftIO $ putStrLn "mxFromListH e"
        pure (Matrix (CLBlasMx mem w' h' d'))

ctypes cls =
  if clblasTypeCStringRep (asProxyTypeOf 0.0 (clProxy cls)) == "double"
    then [cty|double|]
    else [cty|float|]

reshapeMh ::
  forall w h d w2 h2 d2 a.
  (KnownNat w, KnownNat h, KnownNat d, KnownNat w2, KnownNat h2, KnownNat d2, KnownNat (w GHC.TypeLits.* h), KnownNat (w2 GHC.TypeLits.* h2), (w GHC.TypeLits.* h GHC.TypeLits.* d) ~ (w2 GHC.TypeLits.* h2 GHC.TypeLits.* d2)) =>
  Matrix (CLBlasMx a) w h d ->
  Matrix (CLBlasMx a) w2 h2 d2
reshapeMh (Matrix (CLBlasMx mem w h d)) =
  let w' = fromIntegral $ natVal (undefined :: NProxy w2)
      h' = fromIntegral $ natVal (undefined :: NProxy h2)
      d' = fromIntegral $ natVal (undefined :: NProxy d2)
      ow' = fromIntegral $ natVal (undefined :: NProxy w)
      oh' = fromIntegral $ natVal (undefined :: NProxy h)
      od' = fromIntegral $ natVal (undefined :: NProxy d)
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
  let w = fromIntegral $ natVal $ (undefined :: NProxy w)
      h = fromIntegral $ natVal $ (undefined :: NProxy h)
      d = fromIntegral $ natVal $ (undefined :: NProxy d)
   in do
        --liftIO $ putStrLn $ "konst 1 (w,h,d)" <> show (w, h, d)
        cls <- get
        mem <- clCreateBuffer (clContext cls) clMemReadWrite (w * h * d) (clProxy cls)
        --liftIO $ putStrLn "konst af"
        applyFunction (Matrix (CLBlasMx mem w h d)) (Const (fromRational $ toRational v))

constPtr :: (Floating a, Storable a, MonadFail m, MonadIO m, MonadState (CLBlasState a) m) => MxFunction -> m (CLMem a)
constPtr mxf = do
  cls <- get
  clBufferFromList (clContext cls) (clCommandQueue cls) (Prelude.map (fromRational . toRational) (dbs ++ [0.0]))
  where
    dbs = go mxf
    go (Const d) = [d]
    go (Exp f1 f2) = go f1 ++ go f2
    go (Log f1) = go f1
    go (Ln f1) = go f1
    go (Div f1 f2) = go f1 ++ go f2
    go (Mul f1 f2) = go f1 ++ go f2
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
hashFuncToStr mxf = go mxf
  where
    go (Const _) = "C"
    go (Exp f1 f2) = "E" ++ go f1 ++ go f2
    go (Log f1) = "L" ++ go f1
    go (Ln f1) = "l" ++ go f1
    go (Div f1 f2) = "d" ++ go f1 ++ go f2
    go (Mul f1 f2) = "m" ++ go f1 ++ go f2
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

withFuncToCode :: Int -> String -> String -> MxFunction -> (String -> String) -> (Int, String)
withFuncToCode i t vs fx sf =
  let (i', s') = funcToCode i t vs fx
   in (i', sf s')

seqWithFuncToCode :: Int -> String -> String -> MxFunction -> MxFunction -> (String -> String -> String) -> (Int, String)
seqWithFuncToCode i t vs f1 f2 sf =
  let (i1, s1) = funcToCode i t vs f1
      (i2, s2) = funcToCode i1 t vs f2
   in (i2, sf s1 s2)

seqWithFuncToCode3 :: Int -> String -> String -> MxFunction -> MxFunction -> MxFunction -> (String -> String -> String -> String) -> (Int, String)
seqWithFuncToCode3 i t vs f1 f2 f3 sf =
  let (i1, s1) = funcToCode i t vs f1
      (i2, s2) = funcToCode i1 t vs f2
      (i3, s3) = funcToCode i2 t vs f3
   in (i3, sf s1 s2 s3)

funcToCode :: Int -> String -> String -> MxFunction -> (Int, String)
funcToCode i t vs Value = (i, vs)
funcToCode i t vs (Exp f1 f2) = seqWithFuncToCode i t vs f1 f2 (printf "pow(%s,%s)")
funcToCode i t vs (Log f1) = withFuncToCode i t vs f1 (printf "log10(%s)")
funcToCode i t vs (Ln f1) = withFuncToCode i t vs f1 (printf "log(%s)")
funcToCode i t vs (Div f1 f2) = seqWithFuncToCode i t vs f1 f2 (printf "(%s)/(%s)")
funcToCode i t vs (Mul f1 f2) = seqWithFuncToCode i t vs f1 f2 (printf "(%s)*(%s)")
funcToCode i t vs (Add f1 f2) = seqWithFuncToCode i t vs f1 f2 (printf "(%s)+(%s)")
funcToCode i t vs (Sub f1 f2) = seqWithFuncToCode i t vs f1 f2 (printf "(%s)-(%s)")
funcToCode i t _ (Const d) = (i + 1, "consts[" <> show i <> "]")
funcToCode i t vs (Min f1 f2) = seqWithFuncToCode i t vs f1 f2 (printf "min(%s,%s)")
funcToCode i t vs (Max f1 f2) = seqWithFuncToCode i t vs f1 f2 (printf "max(%s,%s)")
funcToCode i t vs (Sinh f) = withFuncToCode i t vs f (printf "sinh(%s)")
funcToCode i t vs (Cosh f) = withFuncToCode i t vs f (printf "cosh(%s)")
funcToCode i t vs (Sqrt f) = withFuncToCode i t vs f (printf "sqrt(%s)")
funcToCode i t vs (Tanh f) = withFuncToCode i t vs f (printf "tanh(%s)")
funcToCode i t vs (If exp f1 f2) =
  let (i', s) = ifExpToCode i exp t vs
   in seqWithFuncToCode i' t vs f1 f2 (printf "((%s)?(%s):(%s))" s)

ifExpToCode :: Int -> IfExp -> String -> String -> (Int, String)
ifExpToCode i (IfGt l r) t vs = seqWithFuncToCode i t vs l r (printf "(%s)>(%s)")
ifExpToCode i (IfLt l r) t vs = seqWithFuncToCode i t vs l r (printf "(%s)<(%s)")
ifExpToCode i (IfEq l r) t vs = seqWithFuncToCode i t vs l r (printf "(%s)==(%s)")
ifExpToCode i (IfNe l r) t vs = seqWithFuncToCode i t vs l r (printf "(%s)!=(%s)")

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
  -- in opencl, add a column at a time
  -- width x depth where width contains the sum of the column
  memo <- clCreateBuffer (clContext cls) clMemReadWrite (fromIntegral (wm * dpt)) (clProxy cls)
  e <-
    clRunKernel
      (clCommandQueue cls)
      kern
      [ CLAPlain (fromIntegral hm :: CInt),
        CLAMem mem1,
        CLAMem memo
      ]
      (fromIntegral wm, Just (fromIntegral dpt), Nothing)
  clWaitForEvent e
  m <- (konst 1.0 :: m (Matrix (CLBlasMx a) 1 w d))
  --liftIO $ putStrLn $ "sumlayers " <> show (Matrix (CLBlasMx memo wm 1 dpt) :: Matrix (CLBlasMx a) w 1 d) <> " with " <> show m
  dense (Matrix (CLBlasMx memo wm 1 dpt) :: Matrix (CLBlasMx a) w 1 d) m

sumFlattenC cls =
  let typ = ctypes cls
   in [cfun| kernel void sumFlatten(int d, int nd, const global $ty:typ * mx, global $ty:typ * out){
              int x = get_global_id(0);
              int y = get_global_id(1);
              int w = get_global_size(0);
              int h = get_global_size(1);
              $ty:typ val = 0;
              for (int i = 0; i < d; ++i){
                val = val + mx[i*w*h+y*w+x];
              }
              for (int i = 0; i < nd; ++i){
                out[i*w*h+y*w+x] = val;
              }
            }
      |]

addLayerC cls =
  let typ = ctypes cls
   in [cfun|kernel void addLayers(int d, const global $ty:typ * mx, const global $ty:typ * v, global $ty:typ * out){
            int x = get_global_id(0);
            int w = get_global_size(0);
            int y = get_global_id(1);
            int h = get_global_size(1);
            for (int k = 0; k < d; ++k){
              int pos = k*w*h + y*w + x;
              out[pos] = mx[pos] + v[k];
            }
          } 
     |]

multLayerC cls =
  let typ = ctypes cls
   in [cfun|kernel void multLayers(int d, const global $ty:typ * mx, const global $ty:typ * v, global $ty:typ * out){
            int x = get_global_id(0);
            int w = get_global_size(0);
            int y = get_global_id(1);
            int h = get_global_size(1);
            for (int k = 0; k < d; ++k){
              int pos = k*w*h + y*w + x;
              out[pos] = mx[pos] * v[k];
            }
          } 
     |]

-- TODO Use a double4 or float4 to process 4 at a time
-- TODO In convolution use local memory for filter and input?
multC cls =
  let typ = ctypes cls
   in [cfun|kernel void multC(const global $ty:typ * mx1, const global $ty:typ * mx2, global $ty:typ * out){
            int x = get_global_id(0);
            int w = get_global_size(0);
            int y = get_global_id(1);
            int h = get_global_size(1);
            int z = get_global_id(2);
            int pos = w*h*z + w * y + x;
            out[pos] = mx1[pos] * mx2[pos];
          } 
     |]

flipC cls =
  let typ = ctypes cls
   in [cfun|kernel void flipMx(int w, int h, int d, int flipx, int flipy, global const $ty:typ * input, global $ty:typ * output){
          int inPosX = get_global_id(0);
          int inPosY = get_global_id(1);

          int outPosX = inPosX;
          if (flipx)
            outPosX = w - outPosX - 1;
          int outPosY = inPosY;
          if (flipy)
            outPosY = h - inPosY - 1;

          int mxSz = w * h;

          for (int i = 0; i < d; i++){
            int mxPos = mxSz * i;
            output[mxPos + outPosY * w + outPosX] = input[mxPos + inPosY * w + inPosX];
          }
        }
      |]

-- add up each column into [col,depth] array
sumC cls =
  let typ = ctypes cls
   in [cfun| kernel void sumC(int h, global const $ty:typ * input, global $ty:typ * output){
        int currWidth = get_global_id(0); // output is depth rows with w columns, pos is a specific width and depth
        int maxWidth = get_global_size(0);
        int currDepth = get_global_id(1);
        $ty:typ val = 0;
        for (int i = 0; i < h ; ++i){
          val += input[currDepth * h * maxWidth + i * maxWidth + currWidth];
        }
        output[currDepth * maxWidth + currWidth] = val;
      }
    |]

{-
  1) Data.Matrix.CLBlasM, check matrix operations, should convolve a matrix 5x5 with 3x3 s1 p0
       expected: [[40.0,41.0,45.0],[40.0,42.0,46.0],[46.0,50.0,55.0]]
        but got: [[50.0,55.0,55.0],[56.0,58.0,60.0],[65.0,70.0,75.0]]
-}

convolveC cls =
  let typ = ctypes cls
   in [cfun|kernel void convolve(int w, int h, int d, global const $ty:typ *input, global const $ty:typ *filter,
                      int fw, int fh, int strideX, int strideY, int padXL, int padXR, int padYT, int padYB,
                      int dialateX, int dialateY,
                      int dialateFX, int dialateFY,
                      int flipFilterX, int flipFilterY,
                      int ow, int oh, global double *output) {
          int outPosX = get_global_id(0);
          int outPosY = get_global_id(1);
          int outPosD = get_global_id(2);
          // number of filters = od, filter size = d
          int od = get_global_size(2);

          $ty:typ val = 0;
          for (int i = 0; i < fw; ++i){
              for (int j = 0; j < fh; ++j){
                int inPosX = outPosX * strideX;
                int inPosY = outPosY * strideY;
                int fxOff = i * (dialateFX + 1);
                int fyOff = j * (dialateFY + 1);
                // am I on a real cell of input
                inPosX = inPosX + fxOff;
                if (inPosX < padXL) {
                  //printf("skip x: %i (%i,%i)", inPosX, outPosX,outPosY);
                  continue;
                }
                inPosY = inPosY + fyOff;
                if (inPosY < padYT) {
                  //printf("skip y: %i (%i,%i)",inPosY, outPosX,outPosY);
                  continue;
                }

                if ((inPosX - padXL) % (dialateX + 1) != 0){
                  //printf("skip diax: %i %i %i",inPosX, padXL, dialateX);  
                  continue;
                }
                  
                if ((inPosY - padYT) % (dialateY + 1) != 0){
                  //printf("skip diay: %i %i %i",inPosY, padYT, dialateY);  
                  continue;
                }

                // I am always on a real cell of filter

                // the real 
                inPosX = (inPosX - padXL) / (dialateX + 1);
                inPosY = (inPosY - padYT) / (dialateY + 1);


                if (inPosX >= w) {
                  //printf("skip x due to width %i %i (%i,%i)",inPosX,w, outPosX, outPosY);
                  continue;
                }
                if (inPosY >= h) {
                  //printf("skip y due to height %i %i (%i,%i) fyoff %i fh %i",inPosY,h, outPosX, outPosY, fyOff, fh);
                  continue;
                }
                //printf("inX %i inY %i for fx %i fy %i ox %i oy %i", inPosX, inPosY, fxOff,fyOff, outPosX, outPosY);

                // flip filter 
                int fPosX = i;
                int fPosY = j;
                if (flipFilterX) fPosX = fw - i - 1;

                if (flipFilterY) fPosY = fh - j - 1;
                
                int inputDepthStart = 0;
                int inputDepthEnd = d;

                // if filter is bigger, subd is the set of layers of the filter to use
                int fdStart = outPosD * d;
                
                for (int id = 0; id < d; ++id){
                  val += (filter[(fdStart+id)*fw*fh + fPosY*fw + fPosX] * input[id * w * h + inPosY * w + inPosX]);
                  
                  //if (outPosX == 1 && outPosY == 1 && (outPosD == 1 || outPosD == 0) && od == 256){
                  //  printf("inX (%i,%i,%i/%i):%f fx (%i,%i,%i):%f  out:(%i,%i,%i) val: %f", inPosX, inPosY, id, d, input[id * w * h + inPosY * w + inPosX], fPosX, fPosY, fdStart + id, filter[(fdStart+id)*fw*fh + fPosY*fw + fPosX], outPosX,outPosY,outPosD, val);
                  //}
                }                    
              }
            }
            //if (outPosD == 1 && outPosY == 1 && outPosX == 1)
            // printf("output (%i,%i,%i): %f",ow,oh,od,val);

            output[outPosD * ow * oh + outPosY * ow + outPosX] = val;
          }
      |]

convolveLayersC cls =
  let typ = ctypes cls
   in [cfun|kernel void convolveLayers(int w, int h, int d, global const $ty:typ *input, global const $ty:typ *filter,
                      int fw, int fh, int strideX, int strideY, int padXL, int padXR, int padYT, int padYB,
                      int dialateX, int dialateY,
                      int dialateFX, int dialateFY,
                      int flipFilterX, int flipFilterY,
                      int ow, int oh, int div, global double *output) {
          int outPosX = get_global_id(0);
          int outPosY = get_global_id(1);
          int outPosD = get_global_id(2);
          // number of filters = od, filter size = d
          int od = get_global_size(2);
          int fd = outPosD;
          int inPosD = outPosD;
          if (!div){
            fd /= d;
            inPosD = inPosD % fd;
          } else {
            inPosD = 0;
          }
          int fdStep = 1;
          int maxFd = fd;
          if (div){
           fdStep = d; 
           maxFd = od * d;
          }



          $ty:typ val = 0;
          for (int i = 0; i < fw; ++i){
              for (int j = 0; j < fh; ++j){
                int inPosX = outPosX * strideX;
                int inPosY = outPosY * strideY;
                int fxOff = i * (dialateFX + 1);
                int fyOff = j * (dialateFY + 1);
                // am I on a real cell of input
                inPosX = inPosX + fxOff;
                if (inPosX < padXL) {
                  //printf("skip x: %i (%i,%i)", inPosX, outPosX,outPosY);
                  continue;
                }
                inPosY = inPosY + fyOff;
                if (inPosY < padYT) {
                  //printf("skip y: %i (%i,%i)",inPosY, outPosX,outPosY);
                  continue;
                }

                if ((inPosX - padXL) % (dialateX + 1) != 0){
                  //printf("skip diax: %i %i %i",inPosX, padXL, dialateX);  
                  continue;
                }
                  
                if ((inPosY - padYT) % (dialateY + 1) != 0){
                  //printf("skip diay: %i %i %i",inPosY, padYT, dialateY);  
                  continue;
                }

                // I am always on a real cell of filter

                // the real 
                inPosX = (inPosX - padXL) / (dialateX + 1);
                inPosY = (inPosY - padYT) / (dialateY + 1);


                if (inPosX >= w) {
                  //printf("skip x due to width %i %i (%i,%i)",inPosX,w, outPosX, outPosY);
                  continue;
                }
                if (inPosY >= h) {
                  //printf("skip y due to height %i %i (%i,%i) fyoff %i fh %i",inPosY,h, outPosX, outPosY, fyOff, fh);
                  continue;
                }
                //printf("inX %i inY %i for fx %i fy %i ox %i oy %i", inPosX, inPosY, fxOff,fyOff, outPosX, outPosY);

                // flip filter 
                int fPosX = i;
                int fPosY = j;
                if (flipFilterX) fPosX = fw - i - 1;

                if (flipFilterY) fPosY = fh - j - 1;
                
                  //if (outPosX == 0 && outPosY == 0)
                    //printf("inX (%i,%i,%i) fx (%i,%i,%i) o (%i,%i) val: %f ", inPosX, inPosY, id, fxOff,fyOff, k, outPosX, outPosY,(filter[k*fw*fh + fPosY*fw + fPosX] * input[id * w * h + inPosY * w + inPosX]));
                
                for (int fdPos = fd; fdPos < maxFd; fdPos += fdStep){     
                  val += (filter[fdPos*fw*fh + fPosY*fw + fPosX] * input[inPosD * w * h + inPosY * w + inPosX]);
                }
            }
            //printf("output (%i,%i,%i):%d",outPosX,outPosY,k,val);
          }
          //if (outPosD == 1 && outPosY == 1 && outPosX == 1)
          //  printf("layerc (%i,%i,%i): %f",ow,oh,od,val);
          output[outPosD * ow * oh + outPosY * ow + outPosX] = val;

          }
      |]
