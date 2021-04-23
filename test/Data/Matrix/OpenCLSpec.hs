module Data.Matrix.OpenCLSpec where

import Data.Matrix.OpenCL
import Test.Hspec

spec :: Spec
spec = do
  describe "OpenCL Tests" $ do
    it "can get number of platforms" $
      do
        n <- clGetPlatforms
        length n `shouldSatisfy` (> 0)
    it
      "can get number of devices"
      $ do
        n <- clGetPlatforms
        d <- clGetDevices (head n)
        length d `shouldSatisfy` (> 0)
    it "can get device info" $
      do
        n <- clGetPlatforms
        d <- clGetDevices (head n)
        di <- mapM clDeviceInfo d
        print di
        length di `shouldSatisfy` (> 0)
        length (clDeviceInfoName (head di)) `shouldSatisfy` (> 0)
