{-# LANGUAGE DataKinds #-}

module ML.NNet.LeakyReluSpec where

import Data.BlasM
import Data.Matrix.CLBlasM
import Data.Proxy
import Foreign.C.Types
import ML.NNet.LeakyRelu
import System.Random
import Test.Hspec
import qualified Data.Vector.Storable as V
mxFromList ls px py pz = mxFromVec (V.fromList ls) px py pz
-- this should learn to detect a circle
spec :: Spec
spec = do
  describe "leakyrelu layer" $ do
    it "should leakyrelu forward" $ do
      v <- withCLGpu (undefined :: Proxy CFloat) $ do
        i <- mxFromList [0, 1, -1, 2, -2, 2, -3, 3] (Proxy :: Proxy 2) (Proxy :: Proxy 2) (Proxy :: Proxy 2)
        (mx, _) <- leakyReluForward (LeakyReluSt 0.02) i
        mxToLists mx
      v `shouldBe` [[[0, 1], [-1.9999999552965164e-2, 2]], [[-3.999999910593033e-2, 2], [-5.999999865889549e-2, 3]]]
