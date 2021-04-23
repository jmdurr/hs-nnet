module ML.NNet.Init.RandomFun where

import Data.Random.Normal
import System.Random

type WeightInitializer g = Int -> Int -> (g -> (Double, g))

kaimingWeights :: (RandomGen g) => WeightInitializer g
kaimingWeights fanIn _ = normalSg 0 (sqrt (2 / fromIntegral fanIn))

heUniformWeights :: (RandomGen g) => WeightInitializer g
heUniformWeights fanIn _ = let val = sqrt (6 / fromIntegral fanIn) in randomR (negate val, val)

kaimingWeightsSg :: (RandomGen g) => Double -> Double -> WeightInitializer g
kaimingWeightsSg mu sigma fanIn _ g =
  let (r, g') = normalSg mu sigma g
   in (r * sqrt (2 / fromIntegral fanIn), g')

glorotUniformWeights :: (RandomGen g) => WeightInitializer g
glorotUniformWeights fanIn fanOut = let val = sqrt 3 * sqrt (6 / fromIntegral (fanIn + fanOut)) in randomR (negate val, val)

normalSg :: (RandomGen g) => Double -> Double -> g -> (Double, g)
normalSg mu sigma =
  normal' (mu, sigma)

-- to do he-et-al we need to know the size of the previous layer?
smallNormal :: (RandomGen g) => (g -> (Double, g))
smallNormal = normalSg 0 0.01

smallUniform :: (RandomGen g) => (g -> (Double, g))
smallUniform = randomR (-0.01, 0.01)
