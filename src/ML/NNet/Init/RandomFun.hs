module ML.NNet.Init.RandomFun where

import Data.Random.Normal
import Debug.Trace
import System.Random

type WeightInitializer g = Int -> Int -> (g -> (Double, g))

kaimingWeights :: (RandomGen g) => WeightInitializer g
kaimingWeights fanIn _ g =
  let (r, g') = normalSg 0 1 g
   in (r * (sqrt 2 / sqrt (fromIntegral fanIn)), g')

normalSg :: (RandomGen g) => Double -> Double -> g -> (Double, g)
normalSg mu sigma =
  normal' (mu, sigma)

-- to do he-et-al we need to know the size of the previous layer?
smallNormal :: (RandomGen g) => (g -> (Double, g))
smallNormal = normalSg 0 0.01

smallUniform :: (RandomGen g) => (g -> (Double, g))
smallUniform = randomR (-0.01, 0.01)
