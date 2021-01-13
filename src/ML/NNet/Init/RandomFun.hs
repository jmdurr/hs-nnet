module ML.NNet.Init.RandomFun where

import System.Random

-- to do he-et-al we need to know the size of the previous layer?
smallUniform :: (RandomGen g) => (g -> (Double, g))
smallUniform = randomR (-0.01, 0.01)
