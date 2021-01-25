{-# LANGUAGE DataKinds #-}

module ML.NNet.FullyConnectedSGDSpec where

import Data.BlasM
import Data.Matrix.CLBlasM
import Data.Proxy
import Foreign.C.Types
import ML.NNet.FullyConnectedSGD
import System.Random
import Test.Hspec

-- this should learn to detect a circle
spec :: Spec
spec =
  describe "small feed forward network" $ do
    it "should initialize weights" $ do
      v <- withCLGpu (undefined :: Proxy CFloat) $ do
        (n, _) <- fullyConnectedSGDInit (Proxy :: Proxy 2) (Proxy :: Proxy 3) (\_ _ -> const (1.0, mkStdGen 2)) (mkStdGen 2)
        mxToLists (ffnW n)
      v `shouldBe` [[[1, 1], [1, 1], [1, 1]]]
    it "should initialize bias" $ do
      v <- withCLGpu (undefined :: Proxy CFloat) $ do
        (n, _) <- fullyConnectedSGDInit (Proxy :: Proxy 2) (Proxy :: Proxy 3) (\_ _ -> const (1.0, mkStdGen 2)) (mkStdGen 2)
        mxToLists (ffnB n)
      v `shouldBe` [[[1], [1], [1]]]

{-
    it "should feed forward" $ do
      v <- withCLGpu (undefined :: Proxy CFloat) $ do
        s <- mxFromList [252, 4, 155, 175] (Proxy :: Proxy 1) (Proxy :: Proxy 4) (Proxy :: Proxy 1)
        b <- mxFromList [1, 1] (Proxy :: Proxy 1) (Proxy :: Proxy 2) (Proxy :: Proxy 1)
        w <- mxFromList [-0.00256, 0.00146, 0.00816, -0.00597, 0.00889, 0.00322, 0.00258, -0.00876] (Proxy :: Proxy 4) (Proxy :: Proxy 2) (Proxy :: Proxy 1)
        i <- konst 0.0
        r <- fst <$> forwardProp (FFN w b i) s
        mxToLists r
      v `shouldBe` [[[0.5807700157165527], [2.1200602054595947]]]
-}
