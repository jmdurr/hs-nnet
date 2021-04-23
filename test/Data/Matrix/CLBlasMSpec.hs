{-# LANGUAGE DataKinds #-}

module Data.Matrix.CLBlasMSpec where

import Control.Concurrent (threadDelay)
import Control.Monad.IO.Class
import Data.BlasM
import Data.Matrix.CLBlasM
import Data.Matrix.OpenCL
import Data.Proxy
import qualified Data.Vector.Storable as V
import Foreign.C.Types
import Test.Hspec

mxFromList ls px py pz = mxFromVec (V.fromList ls) px py pz

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

  it "should perform a largematrix multiplication" $ do
    v <- withCLGpu (undefined :: Proxy CFloat) $ do
      l1 <- mxFromList largeMx1 (undefined :: Proxy 32) (undefined :: Proxy 18) (undefined :: Proxy 1)
      l2 <- mxFromList largeMx2 (undefined :: Proxy 12) (undefined :: Proxy 32) (undefined :: Proxy 1)
      l3 <- dense l1 l2
      mxToVec l3
    v `shouldBe` (V.fromList largeMxMultOut)

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
  it "should convolve a matrix 5x51 with 3x3 s1 p0" $ do
    v <- withCLGpu (undefined :: Proxy CDouble) $ do
      mat <- mxFromList [35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75] (undefined :: Proxy 5) (undefined :: Proxy 5) (undefined :: Proxy 1)
      filt <- mxFromList [0, 1, 0, 0, 0, 0, 0, 0, 0] (undefined :: Proxy 3) (undefined :: Proxy 3) (undefined :: Proxy 1)
      cout <- convolve mat filt (Proxy :: Proxy 1) (undefined :: Proxy 1, undefined :: Proxy 1) (undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0)
      mxToLists cout
    v `shouldBe` [[[40, 41, 45], [40, 42, 46], [46, 50, 55]]]
  it "should convolve a matrix 5x5 with 2x2 s1 p0" $ do
    v <- withCLGpu (undefined :: Proxy CDouble) $ do
      mat <- mxFromList [35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75] (undefined :: Proxy 5) (undefined :: Proxy 5) (undefined :: Proxy 1)
      filt <- mxFromList [0, 1, 0, 0] (undefined :: Proxy 2) (undefined :: Proxy 2) (undefined :: Proxy 1)
      cout <- convolve mat filt (Proxy :: Proxy 1) (undefined :: Proxy 1, undefined :: Proxy 1) (undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0)
      mxToLists cout
    v `shouldBe` [[[40, 41, 45, 50], [40, 42, 46, 52], [46, 50, 55, 55], [52, 56, 58, 60]]]
  it "should convolve a matrix 5x5x5 with 2x2 s1 p0" $ do
    v <- withCLGpu (undefined :: Proxy CDouble) $ do
      mat <- mxFromList [35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75] (undefined :: Proxy 5) (undefined :: Proxy 5) (undefined :: Proxy 5)
      filt <- mxFromList [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0] (undefined :: Proxy 2) (undefined :: Proxy 2) (undefined :: Proxy 5)
      cout <- convolve mat filt (Proxy :: Proxy 1) (undefined :: Proxy 1, undefined :: Proxy 1) (undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0)
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

      cout <- convolve mat filt (Proxy :: Proxy 1) (undefined :: Proxy 2, undefined :: Proxy 2) (undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0)
      mxToLists cout
    v `shouldBe` [[[40, 45, 0], [46, 55, 2], [60, 70, 4]]]
  it "should convolve a matrix 6x6 with 4x4 s2 p1" $ do
    -- (6 - 4 + 2) / 2 + 1 = o - o = 3
    v <- withCLGpu (undefined :: Proxy CDouble) $ do
      -- 6 and 2 gives us 3?
      mat <- mxFromList [35, 40, 41, 45, 50, 0, 40, 40, 42, 46, 52, 1, 42, 46, 50, 55, 55, 2, 48, 52, 56, 58, 60, 3, 56, 60, 65, 70, 75, 4, 5, 6, 7, 8, 9, 10] (undefined :: Proxy 6) (undefined :: Proxy 6) (undefined :: Proxy 1)
      filt <- mxFromList [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (undefined :: Proxy 4) (undefined :: Proxy 4) (undefined :: Proxy 1)
      cout <- convolve mat filt (Proxy :: Proxy 1) (undefined :: Proxy 2, undefined :: Proxy 2) (undefined :: Proxy 1, undefined :: Proxy 1, undefined :: Proxy 1, undefined :: Proxy 1) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0)
      olist <- mxToLists mat
      liftIO $ putStrLn (show olist)
      mxToLists cout
    v `shouldBe` [[[35, 41, 50], [42, 50, 55], [56, 65, 75]]]
  it "should convolve a matrix 5x5 with 3x3 s1 p0 dialation input 1" $ do
    v <- withCLGpu (undefined :: Proxy CDouble) $ do
      mat <- mxFromList [35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75] (undefined :: Proxy 5) (undefined :: Proxy 5) (undefined :: Proxy 1)
      filt <- mxFromList [0, 1, 0, 0, 0, 0, 0, 0, 0] (undefined :: Proxy 3) (undefined :: Proxy 3) (undefined :: Proxy 1)
      cout <- convolve mat filt (Proxy :: Proxy 1) (undefined :: Proxy 1, undefined :: Proxy 1) (undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0, undefined :: Proxy 0) (Proxy :: Proxy 1, Proxy :: Proxy 1) (Proxy :: Proxy 0, Proxy :: Proxy 0)
      mxToLists cout
    v `shouldBe` [[[0, 40, 0, 41, 0, 45, 0], [0, 0, 0, 0, 0, 0, 0], [0, 40, 0, 42, 0, 46, 0], [0, 0, 0, 0, 0, 0, 0], [0, 46, 0, 50, 0, 55, 0], [0, 0, 0, 0, 0, 0, 0], [0, 52, 0, 56, 0, 58, 0]]]
  it "should convolve a matrix 5x5 with 3x3 s1 p1 dialation input 1" $ do
    v <- withCLGpu (undefined :: Proxy CDouble) $ do
      mat <- mxFromList [35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75] (undefined :: Proxy 5) (undefined :: Proxy 5) (undefined :: Proxy 1)
      filt <- mxFromList [0, 1, 0, 0, 0, 0, 0, 0, 0] (undefined :: Proxy 3) (undefined :: Proxy 3) (undefined :: Proxy 1)
      cout <- convolve mat filt (Proxy :: Proxy 1) (undefined :: Proxy 1, undefined :: Proxy 1) (undefined :: Proxy 1, undefined :: Proxy 1, undefined :: Proxy 1, undefined :: Proxy 1) (Proxy :: Proxy 1, Proxy :: Proxy 1) (Proxy :: Proxy 0, Proxy :: Proxy 0)
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
  it "should sum large layers" $ do
    v <- withCLGpu (Proxy :: Proxy CDouble) $ do
      mat <- mxFromList [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4] (Proxy :: Proxy 8) (Proxy :: Proxy 2) (Proxy :: Proxy 2)
      r <- sumLayers mat
      mxToLists r
    v `shouldBe` [[[24]], [[56]]]
  it "should add a number to each layerD" $ do
    v <- withCLGpu (Proxy :: Proxy CDouble) $ do
      mat <- mxFromList [0, 1, 2, 3, 4, 5, 6, 7] (Proxy :: Proxy 2) (Proxy :: Proxy 2) (Proxy :: Proxy 2)
      vec <- mxFromList [1, 2] (Proxy :: Proxy 1) (Proxy :: Proxy 1) (Proxy :: Proxy 2)
      r <- addToAllWithDepth mat vec
      mxToLists r
    v `shouldBe` [[[1, 2], [3, 4]], [[6, 7], [8, 9]]]
  it "should average some matrices" $ do
    v <- withCLGpu (Proxy :: Proxy CDouble) $ do
      mat1 <- mxFromList [0, 1, 3, 10, 7, 5, 6, 10] (Proxy :: Proxy 2) (Proxy :: Proxy 2) (Proxy :: Proxy 2)
      mat2 <- mxFromList [0, 2, 3, 20, 8, 5, 6, 12] (Proxy :: Proxy 2) (Proxy :: Proxy 2) (Proxy :: Proxy 2)
      mat3 <- mxFromList [0, 3, 3, 30, 8, 5, 6, 14] (Proxy :: Proxy 2) (Proxy :: Proxy 2) (Proxy :: Proxy 2)
      mat4 <- mxFromList [0, 4, 3, 40, 7, 5, 6, 16] (Proxy :: Proxy 2) (Proxy :: Proxy 2) (Proxy :: Proxy 2)
      avgMxs [mat1, mat2, mat3, mat4]
    v `shouldBe` 7.75
  it "should average some addavg matrices" $ do
    v <- withCLGpu (Proxy :: Proxy CDouble) $ do
      l1 <- mapM (\l -> mxFromList l (Proxy :: Proxy 1) (Proxy :: Proxy 1) (Proxy :: Proxy 1)) (map (: []) [0.2891650326814946, 0.337566345176627, 0.2846741271485443, 0.3724610531265281, 0.2650271713927996, 0.34274238126306084, 0.3418072378550754, 0.3235384626577655, 0.2654258044524422, 0.24631862468764465, 0.308986574045208, 0.29572408658633026, 0.1774838644228291, 0.24315203791550521, 0.3140711762220271, 0.28376033328916983, 0.28301227281411023, 0.24729906804561885, 0.3304603794488633, 0.29886951770323456, 0.2640479753931653, 0.21487733934502018, 0.32415417195769164, 0.283293220378215, 0.30175953645032316, 0.26910253642352167, 0.28395332435816667, 0.30447086685053215, 0.34426793153920604, 0.3258122427090598, 0.347913717898508, 0.2783041768849305])
      l2 <- mapM (\l -> mxFromList l (Proxy :: Proxy 1) (Proxy :: Proxy 1) (Proxy :: Proxy 1)) (map (: []) [-0.4516795535006138, -0.4504756939538804, -0.45637935462625534, -0.454147003233389, -0.44978169996279904, -0.49705583651786767, -0.45448614512344165, -0.46470038209834064, -0.46521928180973077, -0.45133672076812176, -0.4579483720328839, -0.4490373689411742, -0.4657248756033063, -0.45195075716676136, -0.4540682166878972, -0.4510775769917385, -0.44179777506778917, -0.46481198799562184, -0.4501962165074168, -0.4540009244428773, -0.45066346075188607, -0.48551527470376477, -0.46526052670924845, -0.4546671518473454, -0.49029008389047607, -0.47648344615209, -0.4530688469379986, -0.4731573908991067, -0.46899846461586553, -0.4694070557373539, -0.4569789140854343, -0.4550068086242596])
      r <- (+) <$> avgMxs l1 <*> avgMxs l2
      pure r
    v `shouldSatisfy` (> negate 0.167)
    v `shouldSatisfy` (< negate 0.1668)
  it "should convolve a matrix 5x5x5 with 2x2x10 s2 p2" $ do
    v <- withCLGpu (undefined :: Proxy CDouble) $ do
      mat <- mxFromList [35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75] (undefined :: Proxy 5) (undefined :: Proxy 5) (undefined :: Proxy 5)
      filt <- mxFromList [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0] (undefined :: Proxy 2) (undefined :: Proxy 2) (undefined :: Proxy 10)
      cout <- convolve mat filt (Proxy :: Proxy 2) (undefined :: Proxy 2, undefined :: Proxy 2) (undefined :: Proxy 1, undefined :: Proxy 2, undefined :: Proxy 0, undefined :: Proxy 3) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0)
      mxToLists cout
    v `shouldBe` [[[175.0, 205.0, 250.0, 0], [210.0, 250.0, 275.0, 0.0], [280.0, 325.0, 375.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[175.0, 205.0, 250.0, 0], [210.0, 250.0, 275.0, 0.0], [280.0, 325.0, 375.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
  it "should convolve a matrix 5x5x5 with 2x2x10 s2 p3 di1" $ do
    v <- withCLGpu (undefined :: Proxy CDouble) $ do
      mat <- mxFromList [35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75] (undefined :: Proxy 5) (undefined :: Proxy 5) (undefined :: Proxy 5)
      filt <- mxFromList [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0] (undefined :: Proxy 2) (undefined :: Proxy 2) (undefined :: Proxy 10)
      cout <- convolve mat filt (Proxy :: Proxy 2) (undefined :: Proxy 2, undefined :: Proxy 2) (undefined :: Proxy 1, undefined :: Proxy 2, undefined :: Proxy 0, undefined :: Proxy 3) (Proxy :: Proxy 1, Proxy :: Proxy 1) (Proxy :: Proxy 0, Proxy :: Proxy 0)
      mxToLists cout
    v `shouldBe` [[[175.0, 200.0, 205.0, 225.0, 250.0, 0.0], [200.0, 200.0, 210.0, 230.0, 260.0, 0.0], [210.0, 230.0, 250.0, 275.0, 275.0, 0.0], [240.0, 260.0, 280.0, 290.0, 300.0, 0.0], [280.0, 300.0, 325.0, 350.0, 375.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[175.0, 200.0, 205.0, 225.0, 250.0, 0.0], [200.0, 200.0, 210.0, 230.0, 260.0, 0.0], [210.0, 230.0, 250.0, 275.0, 275.0, 0.0], [240.0, 260.0, 280.0, 290.0, 300.0, 0.0], [280.0, 300.0, 325.0, 350.0, 375.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
  it "should convolve a matrix 5x5x5 with 2x2x10 s2 p4 di1 df1" $ do
    v <- withCLGpu (undefined :: Proxy CDouble) $ do
      mat <- mxFromList [35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75, 35, 40, 41, 45, 50, 40, 40, 42, 46, 52, 42, 46, 50, 55, 55, 48, 52, 56, 58, 60, 56, 60, 65, 70, 75] (undefined :: Proxy 5) (undefined :: Proxy 5) (undefined :: Proxy 5)
      filt <- mxFromList [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0] (undefined :: Proxy 2) (undefined :: Proxy 2) (undefined :: Proxy 10)
      cout <- convolve mat filt (Proxy :: Proxy 2) (undefined :: Proxy 2, undefined :: Proxy 2) (undefined :: Proxy 2, undefined :: Proxy 2, undefined :: Proxy 0, undefined :: Proxy 4) (Proxy :: Proxy 1, Proxy :: Proxy 1) (Proxy :: Proxy 1, Proxy :: Proxy 1)
      mxToLists cout
    v `shouldBe` [[[175.0, 200.0, 205.0, 225.0, 250.0, 0.0], [200.0, 200.0, 210.0, 230.0, 260.0, 0.0], [210.0, 230.0, 250.0, 275.0, 275.0, 0.0], [240.0, 260.0, 280.0, 290.0, 300.0, 0.0], [280.0, 300.0, 325.0, 350.0, 375.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[175.0, 200.0, 205.0, 225.0, 250.0, 0.0], [200.0, 200.0, 210.0, 230.0, 260.0, 0.0], [210.0, 230.0, 250.0, 275.0, 275.0, 0.0], [240.0, 260.0, 280.0, 290.0, 300.0, 0.0], [280.0, 300.0, 325.0, 350.0, 375.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
  it "should backprop convolution" $ do
    (dout, dw, di) <- withCLGpu (Proxy :: Proxy CDouble) $ do
      wgt <- mxFromList [1, 0, -1, 2, 0, -2, 1, 0, -1] (Proxy :: Proxy 3) (Proxy :: Proxy 3) (Proxy :: Proxy 1)
      inp <- mxFromList [1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4] (Proxy :: Proxy 5) (Proxy :: Proxy 6) (Proxy :: Proxy 1)
      cout <- Data.BlasM.convolve inp wgt (Proxy :: Proxy 1) (Proxy :: Proxy 1, Proxy :: Proxy 1) (Proxy :: Proxy 0, Proxy :: Proxy 0, Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0)
      ident <- mxFromList [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] (Proxy :: Proxy 3) (Proxy :: Proxy 4) (Proxy :: Proxy 1)
      mult ident cout
      coutl <- mxToLists cout
      let loss = sum (concat (concat coutl))
      lossmx <- konst 1
      mult lossmx cout
      wgrad <- Data.BlasM.convolveLayers inp lossmx (Proxy :: Proxy 1, Proxy :: Proxy 1) (Proxy :: Proxy 0, Proxy :: Proxy 0, Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0)
      fwgt <- Data.BlasM.flip wgt FlipBoth
      igrad <- Data.BlasM.convolveLayersDy lossmx fwgt (Proxy :: Proxy 1, Proxy :: Proxy 1) (Proxy :: Proxy 2, Proxy :: Proxy 2, Proxy :: Proxy 2, Proxy :: Proxy 2) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0)
      mult igrad inp
      mwgrad <- mxToLists wgrad
      migrad <- mxToLists igrad
      pure (loss, mwgrad, migrad)
    -- rotate kernel, convolve with dy, multiply y
    -- dy has same # of layers as x but we need to output x * fd layers
    -- gradient weights = convolve input with output
    -- dout `shouldBe` 1
    dw `shouldBe` [[[15, 18, 25], [21, 23, 28], [30, 31, 34]]]
    di `shouldBe` [[[1, 1, 0, -1, -1], [3, 3, 0, -3, -3], [4, 4, 0, -4, -4], [4, 4, 0, -4, -4], [3, 3, 0, -3, -3], [1, 1, 0, -1, -1]]]

-- convolve dy with oldin

-- dy, gradient weights

-- TODO test DY convolution

-- (h1,w)T x (h1,h2) -> (w,h2)
--  denseT1 :: (KnownNat w, KnownNat d, KnownNat h1, KnownNat h2) => Matrix mx w h1 d -> Matrix mx h2 h1 d -> m (Matrix mx h2 w d)

-- (h1,w) x (h2,w)T = (h1,w) x (w,h2) -> (h1,h2) -- need to add restrictions?
-- denseT2 :: (KnownNat w, KnownNat d, KnownNat h1, KnownNat h2) => Matrix mx w h1 d -> Matrix mx w h2 d -> m (Matrix mx h2 h1 d)

largeMx1 =
  [ 1,
    8,
    3,
    4,
    8,
    10,
    8,
    7,
    8,
    10,
    8,
    3,
    1,
    7,
    7,
    1,
    3,
    8,
    5,
    5,
    2,
    10,
    8,
    4,
    3,
    1,
    10,
    10,
    6,
    10,
    10,
    2,
    3,
    8,
    1,
    9,
    2,
    9,
    9,
    4,
    3,
    6,
    10,
    2,
    4,
    5,
    3,
    9,
    6,
    3,
    10,
    4,
    2,
    3,
    6,
    5,
    7,
    7,
    7,
    2,
    2,
    3,
    4,
    2,
    8,
    8,
    5,
    9,
    10,
    5,
    2,
    2,
    6,
    10,
    9,
    8,
    4,
    6,
    5,
    8,
    9,
    8,
    6,
    7,
    4,
    1,
    5,
    2,
    10,
    7,
    10,
    2,
    4,
    1,
    10,
    4,
    2,
    1,
    5,
    9,
    8,
    3,
    6,
    7,
    6,
    3,
    3,
    6,
    2,
    6,
    3,
    7,
    6,
    4,
    6,
    10,
    1,
    2,
    7,
    2,
    8,
    2,
    6,
    2,
    9,
    9,
    1,
    10,
    3,
    9,
    3,
    7,
    9,
    6,
    7,
    1,
    5,
    4,
    5,
    2,
    1,
    8,
    1,
    2,
    1,
    6,
    7,
    4,
    1,
    1,
    3,
    8,
    5,
    9,
    5,
    9,
    5,
    4,
    5,
    1,
    5,
    9,
    8,
    8,
    7,
    3,
    7,
    5,
    5,
    3,
    9,
    1,
    7,
    4,
    5,
    9,
    1,
    2,
    2,
    2,
    5,
    3,
    10,
    4,
    10,
    10,
    1,
    7,
    2,
    8,
    3,
    9,
    6,
    1,
    7,
    3,
    1,
    7,
    10,
    3,
    4,
    5,
    2,
    5,
    4,
    2,
    5,
    1,
    2,
    5,
    7,
    7,
    10,
    8,
    1,
    5,
    4,
    3,
    8,
    1,
    5,
    5,
    5,
    5,
    3,
    1,
    3,
    10,
    5,
    9,
    3,
    5,
    2,
    1,
    3,
    3,
    3,
    1,
    3,
    10,
    10,
    1,
    4,
    5,
    1,
    4,
    3,
    1,
    2,
    3,
    10,
    4,
    3,
    8,
    4,
    9,
    10,
    4,
    7,
    5,
    8,
    9,
    3,
    5,
    3,
    1,
    3,
    2,
    1,
    7,
    9,
    7,
    5,
    10,
    6,
    2,
    6,
    8,
    6,
    1,
    3,
    6,
    5,
    5,
    8,
    3,
    6,
    3,
    8,
    6,
    2,
    5,
    10,
    5,
    5,
    7,
    8,
    6,
    3,
    4,
    1,
    8,
    9,
    6,
    2,
    3,
    10,
    10,
    2,
    6,
    10,
    6,
    9,
    7,
    9,
    8,
    1,
    6,
    9,
    9,
    7,
    5,
    9,
    1,
    6,
    9,
    6,
    5,
    2,
    9,
    3,
    4,
    6,
    10,
    1,
    9,
    6,
    8,
    1,
    4,
    1,
    1,
    3,
    7,
    1,
    5,
    7,
    10,
    3,
    2,
    7,
    2,
    8,
    10,
    8,
    9,
    2,
    4,
    5,
    8,
    8,
    3,
    6,
    9,
    2,
    1,
    6,
    8,
    9,
    5,
    6,
    4,
    7,
    3,
    10,
    3,
    4,
    8,
    4,
    10,
    6,
    6,
    4,
    4,
    9,
    1,
    8,
    7,
    8,
    6,
    7,
    3,
    2,
    9,
    3,
    6,
    5,
    8,
    9,
    10,
    3,
    10,
    2,
    9,
    3,
    10,
    9,
    9,
    6,
    8,
    5,
    5,
    5,
    3,
    1,
    7,
    10,
    3,
    8,
    6,
    3,
    6,
    2,
    7,
    3,
    7,
    10,
    7,
    4,
    8,
    8,
    2,
    6,
    5,
    10,
    7,
    1,
    9,
    7,
    10,
    4,
    3,
    2,
    9,
    7,
    8,
    2,
    5,
    10,
    4,
    8,
    6,
    1,
    3,
    7,
    4,
    10,
    2,
    8,
    5,
    8,
    3,
    8,
    1,
    7,
    6,
    7,
    5,
    8,
    1,
    1,
    9,
    1,
    2,
    3,
    3,
    9,
    4,
    7,
    6,
    7,
    4,
    3,
    2,
    10,
    2,
    5,
    5,
    3,
    6,
    4,
    8,
    9,
    5,
    3,
    10,
    4,
    9,
    8,
    7,
    8,
    5,
    7,
    9,
    7,
    7,
    8,
    5,
    10,
    3,
    9,
    9,
    7,
    7,
    10,
    7,
    9,
    10,
    8,
    3,
    3,
    2,
    10,
    10,
    5,
    10,
    1,
    8,
    4,
    7,
    3,
    3,
    8,
    5,
    10,
    5,
    6,
    9,
    2,
    5,
    3,
    2,
    3,
    9,
    7,
    7,
    1,
    9,
    3,
    7,
    7,
    5,
    4,
    5,
    1,
    4,
    1,
    2,
    1,
    9,
    7,
    10,
    8,
    1,
    2,
    7,
    2,
    8,
    7,
    3,
    6,
    3,
    9,
    7,
    9,
    2
  ]

largeMx2 =
  [ 1,
    3,
    4,
    5,
    9,
    10,
    10,
    9,
    10,
    5,
    7,
    4,
    3,
    8,
    2,
    6,
    8,
    4,
    7,
    3,
    5,
    5,
    4,
    4,
    3,
    4,
    1,
    7,
    9,
    1,
    2,
    5,
    10,
    10,
    7,
    4,
    10,
    9,
    3,
    3,
    9,
    5,
    9,
    2,
    7,
    10,
    3,
    1,
    5,
    8,
    1,
    7,
    2,
    1,
    6,
    9,
    6,
    9,
    4,
    10,
    6,
    8,
    7,
    10,
    6,
    4,
    8,
    6,
    4,
    5,
    9,
    7,
    7,
    4,
    10,
    3,
    9,
    1,
    5,
    2,
    7,
    4,
    9,
    8,
    8,
    1,
    6,
    1,
    4,
    5,
    2,
    9,
    7,
    5,
    8,
    1,
    9,
    8,
    9,
    7,
    2,
    5,
    5,
    2,
    3,
    6,
    10,
    7,
    7,
    9,
    1,
    2,
    4,
    6,
    7,
    6,
    9,
    10,
    10,
    10,
    9,
    10,
    2,
    1,
    9,
    7,
    8,
    6,
    10,
    8,
    9,
    8,
    10,
    8,
    8,
    6,
    8,
    7,
    7,
    2,
    8,
    4,
    1,
    9,
    8,
    6,
    3,
    8,
    2,
    10,
    6,
    8,
    8,
    10,
    1,
    1,
    10,
    2,
    6,
    8,
    6,
    2,
    4,
    1,
    8,
    2,
    8,
    8,
    8,
    10,
    9,
    1,
    1,
    7,
    3,
    5,
    7,
    4,
    10,
    7,
    5,
    6,
    9,
    8,
    8,
    6,
    6,
    8,
    10,
    9,
    1,
    10,
    2,
    2,
    5,
    9,
    6,
    4,
    1,
    9,
    1,
    7,
    4,
    4,
    10,
    10,
    2,
    9,
    10,
    8,
    7,
    1,
    2,
    2,
    9,
    3,
    9,
    2,
    8,
    1,
    1,
    1,
    7,
    10,
    10,
    7,
    9,
    9,
    3,
    1,
    5,
    2,
    6,
    4,
    6,
    10,
    1,
    4,
    8,
    1,
    1,
    6,
    1,
    5,
    6,
    4,
    1,
    7,
    8,
    3,
    10,
    3,
    3,
    9,
    3,
    1,
    2,
    9,
    2,
    9,
    6,
    9,
    10,
    4,
    4,
    10,
    4,
    7,
    7,
    9,
    8,
    8,
    6,
    2,
    3,
    9,
    9,
    10,
    10,
    4,
    4,
    7,
    1,
    4,
    10,
    3,
    3,
    9,
    10,
    6,
    5,
    10,
    10,
    9,
    2,
    7,
    8,
    8,
    10,
    4,
    2,
    2,
    7,
    4,
    3,
    6,
    1,
    9,
    3,
    4,
    3,
    6,
    6,
    8,
    4,
    7,
    8,
    9,
    7,
    3,
    7,
    1,
    3,
    9,
    1,
    1,
    10,
    6,
    2,
    3,
    3,
    1,
    8,
    10,
    2,
    2,
    8,
    7,
    7,
    3,
    4,
    1,
    6,
    2,
    7,
    7,
    5,
    4,
    5,
    5,
    8,
    3,
    10,
    2,
    8,
    9,
    10,
    4,
    2,
    3,
    8,
    8,
    3,
    4,
    5,
    7,
    7,
    4,
    3,
    7,
    10,
    8,
    2,
    2,
    8,
    7,
    3,
    6,
    2,
    10,
    9,
    1,
    2,
    3
  ]

largeMxMultOut =
  [ 1199,
    1261,
    1004,
    926,
    1073,
    972,
    1057,
    1006,
    1247,
    1097,
    1267,
    1144,
    1028,
    992,
    873,
    812,
    980,
    855,
    890,
    938,
    1096,
    944,
    986,
    986,
    1223,
    1242,
    916,
    1076,
    1192,
    1094,
    1090,
    1116,
    1269,
    1167,
    1202,
    1186,
    1019,
    909,
    923,
    849,
    974,
    780,
    856,
    986,
    1107,
    909,
    911,
    883,
    925,
    890,
    811,
    771,
    845,
    688,
    792,
    741,
    976,
    854,
    875,
    888,
    983,
    1039,
    925,
    926,
    1046,
    933,
    873,
    1066,
    1264,
    1050,
    968,
    947,
    868,
    881,
    785,
    703,
    856,
    772,
    737,
    861,
    1014,
    794,
    1009,
    812,
    764,
    775,
    780,
    748,
    816,
    712,
    739,
    880,
    917,
    805,
    698,
    745,
    957,
    1021,
    856,
    891,
    954,
    873,
    877,
    967,
    1103,
    953,
    1079,
    944,
    1186,
    1165,
    1130,
    1005,
    1077,
    1085,
    1045,
    1226,
    1349,
    1094,
    1239,
    1202,
    932,
    920,
    844,
    932,
    949,
    825,
    838,
    844,
    1085,
    950,
    926,
    963,
    1061,
    1115,
    1047,
    976,
    1140,
    995,
    986,
    1081,
    1282,
    1112,
    1075,
    1021,
    1154,
    1220,
    1049,
    1027,
    1134,
    1105,
    991,
    1159,
    1356,
    1118,
    1169,
    1139,
    1154,
    1118,
    1048,
    877,
    1055,
    1000,
    986,
    1126,
    1376,
    1121,
    1163,
    1044,
    1042,
    1008,
    919,
    810,
    944,
    870,
    860,
    913,
    1156,
    948,
    1067,
    885,
    1211,
    1214,
    1065,
    1067,
    1112,
    1115,
    1026,
    1207,
    1388,
    1130,
    1150,
    1182,
    1125,
    1175,
    990,
    1111,
    1203,
    1019,
    1000,
    1113,
    1354,
    1096,
    1118,
    1134,
    1058,
    1047,
    909,
    854,
    1014,
    874,
    907,
    896,
    1095,
    968,
    1002,
    941
  ]
