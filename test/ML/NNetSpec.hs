{-# LANGUAGE DataKinds #-}

module ML.NNetSpec where

import Test.Hspec

-- this should learn to detect a circle
spec :: Spec
spec =
  describe "can build a neural network" $
    it "should pend" pending
