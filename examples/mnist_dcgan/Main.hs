{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}

module Main where

import Codec.Picture
import Codec.Picture.Png
import Control.Concurrent (forkIO, yield)
import Control.Concurrent.MVar
import Control.Monad (foldM, foldM_, replicateM, void, when, zipWithM, zipWithM_)
import Control.Monad.IO.Class (liftIO)
import Control.Monad.State.Strict
import Data.BlasM hiding (convolve)
import qualified Data.ByteString as B
import Data.Either
import Data.List (permutations, (!!))
import Data.Matrix.CLBlasM hiding (convolve)
import Data.Proxy
import Data.Serialize
import Data.Text (unpack)
import qualified Data.Vector.Storable as V
import Debug.Trace
import Foreign.C.Types
import ML.NNet
import ML.NNet.Convolve
import ML.NNet.DebugLayer
import ML.NNet.Deconvolve
import ML.NNet.Dropout
import ML.NNet.FullyConnectedSGD
import ML.NNet.GradientMod.Adam
import ML.NNet.GradientMod.Momentum
import ML.NNet.GradientMod.Rate
import ML.NNet.Init.RandomFun
import ML.NNet.LeakyRelu
import ML.NNet.Reshape
import ML.NNet.Sigmoid
import System.IO
import System.Mem
import System.Random
import Text.Printf

-- data in examples/data

getImages :: (BlasM m mx) => Get (m [Matrix mx 28 28 1])
getImages = do
  magic <- getInt32be
  when (magic /= 2051) (fail "magic number invalid")
  numImages <- fromIntegral <$> getInt32be
  numRows <- fromIntegral <$> getInt32be
  numCols <- fromIntegral <$> getInt32be
  when (numRows /= 28) (fail "image rows /= 28")
  when (numCols /= 28) (fail "image cols /= 28")
  -- TODO fix read in less than numImages
  imgs <- replicateM numImages (V.replicateM (numRows * numCols) (((/ 255.0) . fromIntegral) <$> getWord8))
  pure $ mapM (\ims -> mxFromVec ims (Proxy :: Proxy 28) (Proxy :: Proxy 28) (Proxy :: Proxy 1)) imgs

-- unsigned char

type Discriminator m mx blst bist gdb conf gbst g = Network m mx blst bist gdb 28 28 1 1 1 1 conf gbst g

discriminator =
  reshape
    `connect` fullyConnectedSGD (Proxy :: Proxy 784) (Proxy :: Proxy 1024)
    `connect` leakyRelu 0.2 (Proxy :: Proxy 1) (Proxy :: Proxy 1024) (Proxy :: Proxy 1)
    `connect` dropout (mkStdGen 489375213) 0.3
    `connect` fullyConnectedSGD (Proxy :: Proxy 1024) (Proxy :: Proxy 512)
    `connect` leakyRelu 0.2 (Proxy :: Proxy 1) (Proxy :: Proxy 512) (Proxy :: Proxy 1)
    `connect` dropout (mkStdGen 187934743) 0.3
    `connect` fullyConnectedSGD (Proxy :: Proxy 512) (Proxy :: Proxy 256)
    `connect` leakyRelu 0.2 (Proxy :: Proxy 1) (Proxy :: Proxy 256) (Proxy :: Proxy 1)
    `connect` dropout (mkStdGen 397212372) 0.3
    `connect` fullyConnectedSGD (Proxy :: Proxy 256) (Proxy :: Proxy 1)
    `connect` sigmoid (Proxy :: Proxy '(1, 1, 1))

-- `connect` debugLayer "disc - postsigmoid" True True (liftIO . putStrLn . unpack)

{-
  --debugLayer "back disc: " True True (liftIO . putStrLn . unpack)
  convolve (Proxy :: Proxy '(3, 3, 64)) (Proxy :: Proxy '(2, 2)) (Proxy :: Proxy '(0, 1, 0, 1))
    --`connect` debugLayer "back lr2:" True True (liftIO . putStrLn . unpack)
    `connect` leakyRelu 0.2 (Proxy :: Proxy 14) (Proxy :: Proxy 14) (Proxy :: Proxy 64)
    `connect` dropout (mkStdGen 13) 0.4
    --`connect` debugLayer "back cvv:" True True (liftIO . putStrLn . unpack)
    `connect` convolve (Proxy :: Proxy '(3, 3, 64)) (Proxy :: Proxy '(2, 2)) (Proxy :: Proxy '(0, 1, 0, 1))
    --`connect` debugLayer "back lr:" True True (liftIO . putStrLn . unpack)
    `connect` leakyRelu 0.2 (Proxy :: Proxy 7) (Proxy :: Proxy 7) (Proxy :: Proxy 64)
    --`connect` debugLayer "back dropout:" True True (liftIO . putStrLn . unpack)
    `connect` dropout (mkStdGen 17) 0.4
    `connect` reshape
    --`connect` debugLayer "back sgd:" True True (liftIO . putStrLn . unpack)
    `connect` fullyConnectedSGD (Proxy :: Proxy 3136) (Proxy :: Proxy 1)
    `connect` sigmoid (Proxy :: Proxy '(1, 1, 1))
-}
type Generator m mx layerst inputst gradst gradmod gradmodst randgen = Network m mx layerst inputst gradst 1 128 1 28 28 1 gradmod gradmodst randgen

generator =
  fullyConnectedSGD (Proxy :: Proxy 128) (Proxy :: Proxy 256)
    `connect` leakyRelu 0.2 (Proxy :: Proxy 1) (Proxy :: Proxy 256) (Proxy :: Proxy 1)
    `connect` fullyConnectedSGD (Proxy :: Proxy 256) (Proxy :: Proxy 512)
    `connect` leakyRelu 0.2 (Proxy :: Proxy 1) (Proxy :: Proxy 512) (Proxy :: Proxy 1)
    `connect` fullyConnectedSGD (Proxy :: Proxy 512) (Proxy :: Proxy 1024)
    `connect` leakyRelu 0.2 (Proxy :: Proxy 1) (Proxy :: Proxy 1024) (Proxy :: Proxy 1)
    `connect` fullyConnectedSGD (Proxy :: Proxy 1024) (Proxy :: Proxy 784)
    `connect` reshape
    `connect` sigmoid (Proxy :: Proxy '(28, 28, 1))

-- `connect` debugLayer "gen - postsigmoid" True True (liftIO . putStrLn . unpack)

{-
  --debugLayer "before fc" True True (liftIO . putStrLn . unpack)
  -- debugLayer "back fcsgd 1: " True True (liftIO . putStrLn . unpack)
  fullyConnectedSGD (Proxy :: Proxy 100) (Proxy :: Proxy 6272)
    --`connect` debugLayer "after fc" True True (liftIO . putStrLn . unpack)
    `connect` leakyRelu 0.2 (Proxy :: Proxy 1) (Proxy :: Proxy 6272) (Proxy :: Proxy 1)
    `connect` reshape
    --`connect` debugLayer "back decon 0: " True True (liftIO . putStrLn . unpack)
    `connect` deconvolve (Proxy :: Proxy '(4, 4, 64)) (Proxy :: Proxy '(2, 2)) (Proxy :: Proxy '(2, 2, 2, 2))
    --`connect` debugLayer "back relu 2: " True True (liftIO . putStrLn . unpack)
    `connect` leakyRelu 0.2 (Proxy :: Proxy 14) (Proxy :: Proxy 14) (Proxy :: Proxy 64)
    --`connect` debugLayer "back relu 3: " True True (liftIO . putStrLn . unpack)
    --`connect` debugLayer "back decon 2" False True (liftIO $ putStrLn . unpack)
    `connect` deconvolve (Proxy :: Proxy '(4, 4, 32)) (Proxy :: Proxy '(2, 2)) (Proxy :: Proxy '(2, 2, 2, 2))
    --`connect` debugLayer "back decon 1: " True True (liftIO . putStrLn . unpack)
    `connect` leakyRelu 0.2 (Proxy :: Proxy 28) (Proxy :: Proxy 28) (Proxy :: Proxy 32)
    --`connect` debugLayer "back relu 1: " True True (liftIO . putStrLn . unpack)
    `connect` convolve (Proxy :: Proxy '(7, 7, 1)) (Proxy :: Proxy '(1, 1)) (Proxy :: Proxy '(3, 3, 3, 3)) -- this convolve layer is returning crazy values
    --`connect` debugLayer "back convolve 1: " True True (liftIO . putStrLn . unpack)
    --   `connect` debugLayer "presigmoid" True True (liftIO . putStrLn . unpack)
    `connect` sigmoid (Proxy :: Proxy '(28, 28, 1))
-}
--`connect` debugLayer "gen out" True True (liftIO . putStrLn . unpack)

genFake :: (BlasM m mx, RandomGen g) => Network m mx st ist gst 1 128 1 28 28 1 gmod gmodst g -> st -> g -> V.Vector Double -> m (Matrix mx 28 28 1, ist, g)
genFake gen gst g rnds = do
  let (idx, g') = randomR (0, V.length rnds - 130) g
  mx <- mxFromVec (V.slice idx 128 rnds) (Proxy :: Proxy 1) (Proxy :: Proxy 128) (Proxy :: Proxy 1)
  (fake, gist) <- runForward gen gst mx
  pure (fake, gist, g')

-- it would appear that genFakes is causing the memory issue... even if ist is not used
genFakes :: (BlasM m mx, RandomGen g) => Int -> Network m mx st ist gst 1 128 1 28 28 1 gmod gmodst g -> st -> g -> V.Vector Double -> m ([(Matrix mx 28 28 1, ist)], g)
genFakes 0 _ _ g _ = pure ([], g)
genFakes cnt gen gst g rnds = do
  (fake, ist, g') <- genFake gen gst g rnds
  (rs, ge) <- genFakes (cnt -1) gen gst g' rnds
  pure ((fake, ist) : rs, ge)

loadMNist :: (BlasM m mx) => m [Matrix mx 28 28 1]
loadMNist = do
  bs <- liftIO $ B.readFile "./examples/data/train-images.idx3-ubyte"
  either fail id (runGet getImages bs)

randomMNist :: (RandomGen g) => Int -> [Matrix mx 28 28 1] -> g -> ([Matrix mx 28 28 1], g)
randomMNist cnt mxs g =
  let len = length mxs
      go 0 g' = ([], g')
      go l g' =
        let (idx, g1) = randomR (0, (len -1)) g'
            (mxr, gr) = go (l -1) g1
         in ((mxs !! idx) : mxr, gr)
   in go cnt g

saveExImages :: (BlasM m mx) => [Matrix mx 28 28 1] -> m ()
saveExImages images = do
  -- liftIO $ printf "save images iter %d" iter
  let fakes = take 10 images
  vecs <- mapM mxToVec fakes
  zipWithM_ (\i v -> liftIO $ writePng ("./examples/data/mnist-dcgan-out.real." <> show i <> ".png") (mkImg v)) [1 ..] vecs
  where
    mkImg vec = generateImage (\x y -> round (255 * max 0.01 (min 0.99 (vec V.! (y * 28 + x)))) :: Pixel8) 28 28

saveImages :: (RandomGen g, BlasM m mx) => Int -> Generator m mx ga gi ggd ggconf gmod g -> ga -> g -> V.Vector Double -> m ()
saveImages iter gen gst g rands = do
  liftIO $ printf "save images iter %d\n" iter
  (fakes, _) <- genFakes 12 gen gst g rands
  vecs <- mapM (mxToVec . fst) fakes
  zipWithM_ (\i v -> liftIO $ writePng ("./examples/data/mnist-dcgan-out." <> show iter <> "." <> show i <> ".png") (mkImg v)) [1 ..] vecs
  where
    mkImg vec = generateImage (\x y -> round (255 * (vec V.! (y * 28 + x))) :: Pixel8) 28 28

onBatchCount count action cntaction dis gener g rs = go 0 dis gener g rs
  where
    go iter (disc, dst) (gen, gst) g1 (r : reals) = do
      (dst', gst', g') <- cntaction (disc, dst) (gen, gst) g1 (r : [])
      when (iter `mod` count == 0 && iter /= 0) (action (iter `div` count) (disc, dst') (gen, gst') g' [r])
      go (iter + 1) (disc, dst') (gen, gst') g' reals

-- TODO Binary cross entropy and other outputs can be a layer
-- TODO layers should take [Matrix i] [Matrix o] rather than 1 matrix at a time
-- TODO combine runForward,runBackward
-- TODO combine runForward,runBackward,update
-- TODO support logging to TensorFlow format
-- : d*log(out)+(1-d)log(1-out)
crossEntropyError :: (BlasM m mx) => [Matrix mx 1 1 1] -> [Bool] -> m Double
crossEntropyError output desired = do
  -- if desired is 1 then 1 - desired else desired
  -- sum (negative (a*log(p) + (1-a)*log(1-p))) / n
  vals <- mapM (\o -> V.head <$> mxToVec o) output
  -- liftIO $ print vals
  let es = zipWith (\targ out -> (targ * (negate (log out)) + (1 - targ) * negate (log (1 - out)))) (map (\d -> if d then (1.0 - 1e-6) else 1e-6) desired) vals
  pure (sum es)

-- (-d/out + (1-d) / (1-o))
crossEntropyErrorDO :: (BlasM m mx) => Matrix mx 1 1 1 -> Matrix mx 1 1 1 -> m (Matrix mx 1 1 1)
crossEntropyErrorDO output desired = do
  target <- V.head <$> mxToVec desired
  out <- V.head <$> mxToVec output
  let out' = out
  let res = (out' - target) / (out' * (1 - out'))
  --if (out - target <= 0 then 0 else (1-1) / (1*(1-1)) -- infinite)
  -- (0-0) / (0*(1-0)) -- infinite

  -- -999045.65  = (o-1) / (o * (1-o) + 0.0000001)
  -- -999045.65 - 0.099904565 + 1 = o - 999045.65o
  --  -999044.749905 = -999044.65
  -- o = 1.0000001
  -- liftIO $ printf "inp %.2f, des %.2f result %.2f\n" pred label res
  mxFromVec (V.fromList [res]) (Proxy :: Proxy 1) (Proxy :: Proxy 1) (Proxy :: Proxy 1)

{- TODO this process is wrong
  new process:

  for next permutation of input
    for batch in perm
      forward
      back
      accum gradient (average)
    update gradient
    end epoch
-}
processEpoch :: (BlasM m mx, RandomGen g) => Int -> (Discriminator m mx a i gd gconf amod g, a) -> (Generator m mx ga gi ggd ggconf gmod g, ga) -> g -> [Matrix mx 28 28 1] -> V.Vector Double -> m (a, ga, g)
processEpoch batchSize (disc, dst) (gen, gst) g allrs rands = do
  liftIO $ printf "--epoch--\n"
  (gr, g') <- go (take batchSize allrs) (drop batchSize allrs) Nothing g
  case gr of
    Nothing -> fail "no gradients from epoch"
    Just (dg, gg) -> do
      dg' <- divideGradients disc dg (length allrs * 2)
      gg' <- divideGradients gen gg (length allrs)
      gst' <- runUpdate gen gst gg'
      dst' <- runUpdate disc dst dg'
      liftIO $ printf "--end epoch--\n"
      pure (dst', gst', g')
  where
    go _ [] grads g0 = pure (grads, g0)
    go batch remains grads g0 = do
      let r = batch

      --process (disc,dst) (gen,gst) g0 allrs
      (fakesAndInputs, g0') <- genFakes (length r) gen gst g0 rands
      discFakes <- mapM (runForward disc dst . fst) fakesAndInputs
      discReals <- mapM (runForward disc dst) r

      let fakeTarget = replicate (length discFakes) False
      let realTarget = replicate (length discReals) True
      let ganTarget = replicate (length discFakes) True

      derr <- crossEntropyError (map fst discFakes ++ map fst discReals) (fakeTarget ++ realTarget)
      gerr <- crossEntropyError (map fst discFakes) ganTarget

      zeros <- replicateM (length discFakes) (konst 1e-6)
      ones <- replicateM (length discReals) (konst (1.0 - 1e-6))

      deriv <- zipWithM crossEntropyErrorDO (map fst discFakes ++ map fst discReals) (zeros ++ ones)
      genDeriv <- zipWithM crossEntropyErrorDO (map fst discFakes) ones

      bpropd <- avgMxs deriv
      bpropg <- avgMxs genDeriv
      liftIO $ printf "%.2f,%.2f,%.2f,%.2f\n" derr gerr bpropd bpropg

      discGrad <- zipWithM (runBackward disc dst) (map snd discFakes ++ map snd discReals) deriv
      discGenGrad <- zipWithM (runBackward disc dst) (map snd discFakes) genDeriv
      genGrad <- zipWithM (runBackward gen gst) (map snd fakesAndInputs) (map fst discGenGrad)
      -- dst' <- foldM (\dstu grad -> runUpdate disc dstu grad) dst (map snd fakeGrad ++ map snd realGrad)
      -- gst' <- foldM (\gstu grad -> runUpdate gen gstu grad) gst (map snd genGrad)
      dga <- foldl1 (\g1 g2 -> g1 >>= \g1' -> g2 >>= \g2' -> accumulateGradients disc g1' g2') (map (pure . snd) discGrad)
      dga' <- maybe (pure dga) (accumulateGradients disc dga . fst) grads
      gga <- foldl1 (\g1 g2 -> g1 >>= \g1' -> g2 >>= \g2' -> accumulateGradients gen g1' g2') (map (pure . snd) genGrad)
      gga' <- maybe (pure gga) (accumulateGradients gen gga . snd) grads
      liftIO $ performGC >> yield
      go (take batchSize remains) (drop batchSize remains) (Just (dga', gga')) g0'

--liftIO $ performMajorGC
--pure (dst,gst,g1)

-- real error is result - 1

-- there is a GPU memory leak in here, turning on heap profiling causes it to go away
-- Someone on IRC mentioned heap profiling can break floating things out optimizations
-- Need to figure out where that might be occuring... but compiling with
-- -O (no optimizations) does not help?
main :: IO ()
main = do
  g1 <- newStdGen
  withCLGpu (Proxy :: Proxy CFloat) $ do
    liftIO $ putStrLn "load mnist"
    mnist <- loadMNist
    -- make same random numbers to sample
    liftIO $ putStrLn "make some rands"
    let v = V.fromList (take 200000 (randomRs (0 :: Double, 1) g1))
    liftIO $ putStrLn "save real images"
    saveExImages mnist
    liftIO $ printf "loaded %d images\n" (length mnist)
    liftIO $ putStrLn "init discriminator"
    (disc, dst, g) <- initNetwork g1 glorotUniformWeights discriminator (Adam 0.0002 0.5 0.9999)
    liftIO $ putStrLn "init generator"
    (gen, gst, g') <- initNetwork g glorotUniformWeights generator (Adam 0.0002 0.5 0.9999)
    liftIO $ putStrLn "save initial generations"
    saveImages (0 :: Int) gen gst g' v
    liftIO $ putStrLn "start training"
    void $
      foldM
        ( \(dst', gst', ge, iter) perm -> do
            (dste, gste, ge') <- processEpoch 100 (disc, dst') (gen, gst') ge perm v
            saveImages iter gen gste ge' v
            pure (dste, gste, ge', iter + 1)
        )
        (dst, gst, g', 0)
        (take 250 $ permutations mnist)
