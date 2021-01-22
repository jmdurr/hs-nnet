{-# LANGUAGE DataKinds #-}

module ML.NNet.LeakyReluSpec where

import Data.BlasM
import Data.Matrix.CLBlasM
import Data.Proxy
import Foreign.C.Types
import ML.NNet.LeakyRelu
import System.Random
import Test.Hspec

-- this should learn to detect a circle
spec :: Spec
spec = do
  describe "leakyrelu layer" $ do
    it "should leakyrelu forward" $ do
      v <- withCLGpu (undefined :: Proxy CFloat) $ do
        i <- mxFromList [0, 1, -1, 2, -2, 2, -3, 3] (Proxy :: Proxy 2) (Proxy :: Proxy 2) (Proxy :: Proxy 2)
        (mx, _) <- leakyReluForward (LeakyReluSt 0.02) i
        mxToLists mx
      v `shouldBe` [[[0, 1], [-0.20000000298023224, 2]], [[-0.4000000059604645, 2], [-0.6000000238418579, 3]]]
