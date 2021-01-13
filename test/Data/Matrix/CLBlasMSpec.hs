{-# LANGUAGE DataKinds #-}

module Data.Matrix.CLBlasMSpec where

import Control.Concurrent (threadDelay)
import Control.Monad.IO.Class
import Data.BlasM
import Data.Matrix.CLBlasM
import Data.Matrix.OpenCL
import Data.Proxy
import Foreign.C.Types
import Test.Hspec

spec :: Spec
spec = describe "check matrix operations" $ do
  it "should perform an outer product" $ do
    v <- withCLGpu (undefined :: Proxy CFloat) $ do
      l1 <- mxFromList [1, 2, 3, 4, 5] (undefined :: Proxy 1) (undefined :: Proxy 5) (undefined :: Proxy 1)
      l2 <- mxFromList [6, 7, 8, 9, 1, 2, 3, 4, 5] (undefined :: Proxy 1) (undefined :: Proxy 9) (undefined :: Proxy 1)
      l3 <- outer l1 l2
      mxToLists l3
    v `shouldBe` [[[6, 7, 8, 9, 1, 2, 3, 4, 5], [12, 14, 16, 18, 2, 4, 6, 8, 10], [18, 21, 24, 27, 3, 6, 9, 12, 15], [24, 28, 32, 36, 4, 8, 12, 16, 20], [30, 35, 40, 45, 5, 10, 15, 20, 25]]]
  it "should perform an outer product 3d" $ do
    v <- withCLGpu (undefined :: Proxy CFloat) $ do
      l1 <- mxFromList [1, 2, 3, 4, 5] (undefined :: Proxy 1) (undefined :: Proxy 5) (undefined :: Proxy 1)
      l2 <- mxFromList [6, 7, 8, 9, 1, 2, 3, 4, 5] (undefined :: Proxy 1) (undefined :: Proxy 9) (undefined :: Proxy 1)
      l3 <- outer l1 l2
      mxToLists l3
    v `shouldBe` [[[6, 7, 8, 9, 1, 2, 3, 4, 5], [12, 14, 16, 18, 2, 4, 6, 8, 10], [18, 21, 24, 27, 3, 6, 9, 12, 15], [24, 28, 32, 36, 4, 8, 12, 16, 20], [30, 35, 40, 45, 5, 10, 15, 20, 25]]]
  it "should perform a matrix multiplication" $ do
    v <- withCLGpu (undefined :: Proxy CFloat) $ do
      l1 <- mxFromList [1, 2, 3, 4, 5, 6] (undefined :: Proxy 3) (undefined :: Proxy 2) (undefined :: Proxy 1)
      l2 <- mxFromList [9, 8, 7] (undefined :: Proxy 1) (undefined :: Proxy 3) (undefined :: Proxy 1)
      l3 <- dense l1 l2
      mxToLists l3
    v `shouldBe` [[[46], [118]]]
  it "should perform T1 matrix multiplication" $ do
    v <- withCLGpu (undefined :: Proxy CFloat) $ do
      l1 <- mxFromList [1, 4, 2, 5, 3, 6] (undefined :: Proxy 2) (undefined :: Proxy 3) (undefined :: Proxy 1)
      l2 <- mxFromList [9, 8, 7] (undefined :: Proxy 1) (undefined :: Proxy 3) (undefined :: Proxy 1)
      l3 <- denseT1 l1 l2
      mxToLists l3
    v `shouldBe` [[[46], [118]]]
  it "should perform T1 matrix multiplication 3d" $ do
    v <- withCLGpu (undefined :: Proxy CFloat) $ do
      l1 <- mxFromList [1, 4, 2, 5, 3, 6, 1, 4, 2, 5, 3, 6] (undefined :: Proxy 2) (undefined :: Proxy 3) (undefined :: Proxy 2)
      l2 <- mxFromList [9, 8, 7, 9, 8, 7] (undefined :: Proxy 1) (undefined :: Proxy 3) (undefined :: Proxy 2)
      l3 <- denseT1 l1 l2
      mxToLists l3
    v `shouldBe` [[[46], [118]], [[46], [118]]]
  it "should perform T2 matrix multiplication" $ do
    v <- withCLGpu (undefined :: Proxy CFloat) $ do
      l1 <- mxFromList [1, 2, 3, 4, 5, 6] (undefined :: Proxy 3) (undefined :: Proxy 2) (undefined :: Proxy 1) -- (2,3) x (3,1) -> (2,1)
      l2 <- mxFromList [9, 8, 7] (undefined :: Proxy 3) (undefined :: Proxy 1) (undefined :: Proxy 1)
      --   denseT2 :: (KnownNat w, KnownNat d, KnownNat h1, KnownNat h2) => Matrix mx w h1 d -> Matrix mx w h2 d -> m (Matrix mx h2 h1 d)
      -- w=3,h1=2,h2=1 Mx 1 2 -- 2 rows, 1 col
      l3 <- denseT2 l1 l2
      --liftIO $ putStrLn (show l3)
      mxToLists l3
    v `shouldBe` [[[46], [118]]]
  it "should perform T2 matrix multiplication 3d" $ do
    v <- withCLGpu (undefined :: Proxy CFloat) $ do
      l1 <- mxFromList [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6] (undefined :: Proxy 3) (undefined :: Proxy 2) (undefined :: Proxy 2) -- (2,3) x (3,1) -> (2,1)
      l2 <- mxFromList [9, 8, 7, 9, 8, 7] (undefined :: Proxy 3) (undefined :: Proxy 1) (undefined :: Proxy 2)
      --   denseT2 :: (KnownNat w, KnownNat d, KnownNat h1, KnownNat h2) => Matrix mx w h1 d -> Matrix mx w h2 d -> m (Matrix mx h2 h1 d)
      -- w=3,h1=2,h2=1 Mx 1 2 -- 2 rows, 1 col
      l3 <- denseT2 l1 l2
      --liftIO $ putStrLn (show l3)
      mxToLists l3
    v `shouldBe` [[[46], [118]], [[46], [118]]]
  it "should scale matrix" $ do
    v <- withCLGpu (undefined :: Proxy CFloat) $ do
      l1 <- mxFromList [1, 2, 3, 4, 5, 6] (undefined :: Proxy 3) (undefined :: Proxy 2) (undefined :: Proxy 1) -- (2,3) x (3,1) -> (2,1)
      l3 <- scale l1 2.0
      mxToLists l3
    v `shouldBe` [[[2, 4, 6], [8, 10, 12]]]
  it "should perform hadamard multiplication 2 matrices 3d" $ do
    v <- withCLGpu (undefined :: Proxy CFloat) $ do
      l1 <- mxFromList [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6] (undefined :: Proxy 3) (undefined :: Proxy 2) (undefined :: Proxy 2)
      l2 <- mxFromList [10, 11, 12, 13, 14, 15, 10, 11, 12, 13, 14, 15] (undefined :: Proxy 3) (undefined :: Proxy 2) (undefined :: Proxy 2)
      l3 <- mult l1 l2
      mxToLists l3
    v `shouldBe` [[[10, 22, 36], [52, 70, 90]], [[10, 22, 36], [52, 70, 90]]]
  it "should scale 3d matrix" $ do
    v <- withCLGpu (Proxy :: Proxy CDouble) $ do
      l1 <- mxFromList [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6] (undefined :: Proxy 3) (undefined :: Proxy 2) (undefined :: Proxy 2)
      l3 <- scale l1 2.0
      mxToLists l3
    v `shouldBe` [[[2, 4, 6], [8, 10, 12]], [[2, 4, 6], [8, 10, 12]]]
  it "should add two matrices" $ do
    v <- withCLGpu (undefined :: Proxy CFloat) $ do
      l1 <- mxFromList [1, 2, 3, 4, 5, 6] (undefined :: Proxy 3) (undefined :: Proxy 2) (undefined :: Proxy 1)
      l2 <- mxFromList [10, 11, 12, 13, 14, 15] (undefined :: Proxy 3) (undefined :: Proxy 2) (undefined :: Proxy 1)
      l3 <- add l1 l2
      mxToLists l3
    v `shouldBe` [[[11, 13, 15], [17, 19, 21]]]
  it "should add two matrices 3d" $ do
    v <- withCLGpu (undefined :: Proxy CFloat) $ do
      l1 <- mxFromList [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6] (undefined :: Proxy 3) (undefined :: Proxy 2) (undefined :: Proxy 2)
      l2 <- mxFromList [10, 11, 12, 13, 14, 15, 10, 11, 12, 13, 14, 15] (undefined :: Proxy 3) (undefined :: Proxy 2) (undefined :: Proxy 2)
      l3 <- add l1 l2
      mxToLists l3
    v `shouldBe` [[[11, 13, 15], [17, 19, 21]], [[11, 13, 15], [17, 19, 21]]]
  it "should subtract two matrices" $ do
    v <- withCLGpu (undefined :: Proxy CFloat) $ do
      l1 <- mxFromList [1, 2, 3, 4, 5, 6] (undefined :: Proxy 3) (undefined :: Proxy 2) (undefined :: Proxy 1)
      l2 <- mxFromList [10, 11, 14, 9, 8, 7] (undefined :: Proxy 3) (undefined :: Proxy 2) (undefined :: Proxy 1)
      l3 <- subtractM l2 l1
      mxToLists l3
    v `shouldBe` [[[9, 9, 11], [5, 3, 1]]]
  it "should convolve a matrix 5x5 with 3x3 s1 p0" $ do
    v <- withCLGpu (undefined :: Proxy CDouble) $ do
      mat <- mxFromList [35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75] (undefined :: Proxy 5) (undefined :: Proxy 5) (undefined :: Proxy 1)
      filt <- mxFromList [0, 1, 0, 0, 0, 0, 0, 0, 0] (undefined :: Proxy 3) (undefined :: Proxy 3) (undefined :: Proxy 1)
      cout <- convolve mat filt (Proxy :: Proxy 1) (undefined :: Proxy 1, undefined :: Proxy 1) (undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0) False
      mxToLists cout
    v `shouldBe` [[[40, 41, 45], [40, 42, 46], [46, 50, 55]]]
  it "should convolve a matrix 5x5 with 2x2 s1 p0" $ do
    v <- withCLGpu (undefined :: Proxy CDouble) $ do
      mat <- mxFromList [35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75] (undefined :: Proxy 5) (undefined :: Proxy 5) (undefined :: Proxy 1)
      filt <- mxFromList [0, 1, 0, 0] (undefined :: Proxy 2) (undefined :: Proxy 2) (undefined :: Proxy 1)
      cout <- convolve mat filt (Proxy :: Proxy 1) (undefined :: Proxy 1, undefined :: Proxy 1) (undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0) False
      mxToLists cout
    v `shouldBe` [[[40, 41, 45, 50], [40, 42, 46, 52], [46, 50, 55, 55], [52, 56, 58, 60]]]
  it "should convolve a matrix 5x5x5 with 2x2 s1 p0" $ do
    v <- withCLGpu (undefined :: Proxy CDouble) $ do
      mat <- mxFromList [35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75] (undefined :: Proxy 5) (undefined :: Proxy 5) (undefined :: Proxy 5)
      filt <- mxFromList [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0] (undefined :: Proxy 2) (undefined :: Proxy 2) (undefined :: Proxy 5)
      liftIO $ threadDelay 30000000
      cout <- convolve mat filt (Proxy :: Proxy 1) (undefined :: Proxy 1, undefined :: Proxy 1) (undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0) False
      mxToLists cout
    v `shouldBe` [[[200.0, 205.0, 225.0, 250.0], [200.0, 210.0, 230.0, 260.0], [230.0, 250.0, 275.0, 275.0], [260.0, 280.0, 290.0, 300.0]]]
  it "should convolve a matrix 6x6 with 2x2 s2 p0" $ do
    v <- withCLGpu (undefined :: Proxy CDouble) $ do
      -- 6 and 2 gives us 3?
      mat <- mxFromList [35, 40, 41, 45, 50, 0, 40, 40, 42, 46, 52, 1, 42, 46, 50, 55, 55, 2, 48, 52, 56, 58, 60, 3, 56, 60, 65, 70, 75, 4, 5, 6, 7, 8, 9, 10] (undefined :: Proxy 6) (undefined :: Proxy 6) (undefined :: Proxy 1)
      filt <- mxFromList [0, 1, 0, 0] (undefined :: Proxy 2) (undefined :: Proxy 2) (undefined :: Proxy 1)
      -- 6 - 2 + 0 + 0 / 2 + 1 = 3
      -- Mod ((w + ((w - 1) GHC.TypeLits.* dx)) + pxl + pxr - (fw + ((fw - 1) GHC.TypeLits.* dfx)) + 1) sx ~ 0,
      -- 6 + 0 + 0 + 0 - fw

      cout <- convolve mat filt (Proxy :: Proxy 1) (undefined :: Proxy 2, undefined :: Proxy 2) (undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0) False
      mxToLists cout
    v `shouldBe` [[[40, 45, 0], [46, 55, 2], [60, 70, 4]]]
  it "should convolve a matrix 6x6 with 4x4 s2 p1" $ do
    -- (6 - 4 + 2) / 2 + 1 = o - o = 3
    v <- withCLGpu (undefined :: Proxy CDouble) $ do
      -- 6 and 2 gives us 3?
      mat <- mxFromList [35, 40, 41, 45, 50, 0, 40, 40, 42, 46, 52, 1, 42, 46, 50, 55, 55, 2, 48, 52, 56, 58, 60, 3, 56, 60, 65, 70, 75, 4, 5, 6, 7, 8, 9, 10] (undefined :: Proxy 6) (undefined :: Proxy 6) (undefined :: Proxy 1)
      filt <- mxFromList [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] (undefined :: Proxy 4) (undefined :: Proxy 4) (undefined :: Proxy 1)
      cout <- convolve mat filt (Proxy :: Proxy 1) (undefined :: Proxy 2, undefined :: Proxy 2) (undefined :: Proxy 1, undefined :: Proxy 1, undefined :: Proxy 1, undefined :: Proxy 1) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0) False
      mxToLists cout
    v `shouldBe` [[[35, 41, 50], [42, 50, 55], [56, 65, 75]]]
  it "should convolve a matrix 5x5 with 3x3 s1 p0 dialation input 1" $ do
    v <- withCLGpu (undefined :: Proxy CDouble) $ do
      mat <- mxFromList [35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75] (undefined :: Proxy 5) (undefined :: Proxy 5) (undefined :: Proxy 1)
      filt <- mxFromList [0, 1, 0, 0, 0, 0, 0, 0, 0] (undefined :: Proxy 3) (undefined :: Proxy 3) (undefined :: Proxy 1)
      cout <- convolve mat filt (Proxy :: Proxy 1) (undefined :: Proxy 1, undefined :: Proxy 1) (undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0) (Proxy :: Proxy 1, Proxy :: Proxy 1) (Proxy :: Proxy 0, Proxy :: Proxy 0) False
      mxToLists cout
    v `shouldBe` [[[0, 40, 0, 41, 0, 45, 0], [0, 0, 0, 0, 0, 0, 0], [0, 40, 0, 42, 0, 46, 0], [0, 0, 0, 0, 0, 0, 0], [0, 46, 0, 50, 0, 55, 0], [0, 0, 0, 0, 0, 0, 0], [0, 52, 0, 56, 0, 58, 0]]]
  it "should convolve a matrix 5x5 with 3x3 s1 p1 dialation input 1" $ do
    v <- withCLGpu (undefined :: Proxy CDouble) $ do
      mat <- mxFromList [35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75] (undefined :: Proxy 5) (undefined :: Proxy 5) (undefined :: Proxy 1)
      filt <- mxFromList [0, 1, 0, 0, 0, 0, 0, 0, 0] (undefined :: Proxy 3) (undefined :: Proxy 3) (undefined :: Proxy 1)
      cout <- convolve mat filt (Proxy :: Proxy 1) (undefined :: Proxy 1, undefined :: Proxy 1) (undefined :: Proxy 1, undefined :: Proxy 1, undefined :: Proxy 1, undefined :: Proxy 1) (Proxy :: Proxy 1, Proxy :: Proxy 1) (Proxy :: Proxy 0, Proxy :: Proxy 0) False
      mxToLists cout
    v `shouldBe` [[[0, 0, 0, 0, 0, 0, 0, 0, 0], [35, 0, 40, 0, 41, 0, 45, 0, 50], [0, 0, 0, 0, 0, 0, 0, 0, 0], [40, 0, 40, 0, 42, 0, 46, 0, 52], [0, 0, 0, 0, 0, 0, 0, 0, 0], [42, 0, 46, 0, 50, 0, 55, 0, 55], [0, 0, 0, 0, 0, 0, 0, 0, 0], [48, 0, 52, 0, 56, 0, 58, 0, 60], [0, 0, 0, 0, 0, 0, 0, 0, 0]]]

  it "should flip a matrix over x axis" $ do
    v <- withCLGpu (Proxy :: Proxy CDouble) $ do
      mat <- mxFromList [1, 2, 3, 4, 5, 10, 11, 12, 13, 14] (Proxy :: Proxy 5) (Proxy :: Proxy 1) (Proxy :: Proxy 2)
      flp <- Data.BlasM.flip mat FlipX
      mxToLists flp
    v `shouldBe` [[[5, 4, 3, 2, 1]], [[14.0, 13.0, 12.0, 11.0, 10.0]]]
  it "should sqrt a matrix values" $ do
    v <- withCLGpu (Proxy :: Proxy CDouble) $ do
      mat <- mxFromList [1, 64, 4, 9, 16, 25, 36, 49] (Proxy :: Proxy 2) (Proxy :: Proxy 2) (Proxy :: Proxy 2)
      r <- applyFunction mat (Sqrt Value)
      mxToLists r
    v `shouldBe` [[[1, 8], [2, 3]], [[4, 5], [6, 7]]]
  it "should sum layers" $ do
    v <- withCLGpu (Proxy :: Proxy CDouble) $ do
      mat <- mxFromList [0, 1, 2, 3, 4, 5, 6, 7] (Proxy :: Proxy 2) (Proxy :: Proxy 2) (Proxy :: Proxy 2)
      r <- sumLayers mat
      mxToLists r
    v `shouldBe` [[[6]], [[22]]]

-- TODO test convolution with input dialation
-- TODO test convolution with filter dialation
-- TODO test convolution with both input and filter dialation
-- TODO test convolution with dialations and padding and stride

-- (h1,w)T x (h1,h2) -> (w,h2)
--  denseT1 :: (KnownNat w, KnownNat d, KnownNat h1, KnownNat h2) => Matrix mx w h1 d -> Matrix mx h2 h1 d -> m (Matrix mx h2 w d)

-- (h1,w) x (h2,w)T = (h1,w) x (w,h2) -> (h1,h2) -- need to add restrictions?
-- denseT2 :: (KnownNat w, KnownNat d, KnownNat h1, KnownNat h2) => Matrix mx w h1 d -> Matrix mx w h2 d -> m (Matrix mx h2 h1 d)
