{-# LANGUAGE DataKinds #-}

module ML.NNet.ReluSpec where

import Data.BlasM
import Data.Matrix.CLBlasM
import Data.Proxy
import Foreign.C.Types
import ML.NNet.Relu
import System.Random
import Test.Hspec
import qualified Data.Vector.Storable as V
mxFromList ls px py pz = mxFromVec (V.fromList ls) px py pz
-- this should learn to detect a circle
spec :: Spec
spec = do
  describe "relu layer" $ do
    it "should relu forward" $ do
      v <- withCLGpu (undefined :: Proxy CFloat) $ do
        i <- mxFromList [0, 1, -1, 2, -2, 2, -3, 3] (Proxy :: Proxy 2) (Proxy :: Proxy 2) (Proxy :: Proxy 2)
        (mx, _) <- reluForward ReluSt i
        mxToLists mx
      v `shouldBe` [[[0, 1], [0, 2]], [[0, 2], [0, 3]]]
