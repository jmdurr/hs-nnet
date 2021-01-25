{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module ML.NNet.LeakyRelu where

import Data.BlasM
import Data.Proxy
import Debug.Trace
import GHC.TypeLits
import ML.NNet
import ML.NNet.Init.RandomFun
import System.Random

data LeakyReluSt = LeakyReluSt Double

type LeakyReluIn mx w h d = Matrix mx w h d

data LeakyReluG = LeakyReluG

leakyReluForward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => LeakyReluSt -> Matrix mx w h d -> m (Matrix mx w h d, LeakyReluIn mx w h d)
leakyReluForward (LeakyReluSt sc) mx = do
  mx' <- mx `applyFunction` (If (IfLt Value (Const 0.0)) (Mul (Const sc) Value) Value) -- 0 if input is < 0 else input
  pure (mx', mx)

leakyReluBackward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => LeakyReluSt -> LeakyReluIn mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, LeakyReluG)
leakyReluBackward (LeakyReluSt sc) inp mx = do
  -- if input is < 0 then 0 else 1
  -- max 0 Value / Value
  -- input is stored in LeakyRelu
  -- v <- applyFunction inp (Max (Const 0.0) Value `Div` Max (Const 0.00000000000001) Value) -- if input is < 0 make it 0, divide by original value to make 1 but if it is 0 we need it to be 1...?
  v <- applyFunction inp (If (IfGt Value (Const 0.0)) (Const 1.0) (Const sc)) -- run this on input to generate 0 or 1
  v' <- mult mx v
  -- multiply 1 for 1 with mx
  pure (v', LeakyReluG)

leakyReluG :: Monad m => [LeakyReluG] -> m LeakyReluG
leakyReluG _ = pure LeakyReluG

leakyReluU :: (Monad m) => conf -> LeakyReluSt -> LeakyReluG -> m LeakyReluSt
leakyReluU _ st _ = pure st

defLeakyRelu :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g) => Proxy w -> Proxy h -> Proxy d -> Layer m mx LeakyReluSt (LeakyReluIn mx w h d) LeakyReluG w h d w h d conf mod g
defLeakyRelu w h d = Layer leakyReluForward leakyReluBackward leakyReluG leakyReluU (leakyReluInit 0.02)

leakyRelu :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g) => Double -> Proxy w -> Proxy h -> Proxy d -> Layer m mx LeakyReluSt (LeakyReluIn mx w h d) LeakyReluG w h d w h d conf mod g
leakyRelu sc w h d = Layer leakyReluForward leakyReluBackward leakyReluG leakyReluU (leakyReluInit sc)

leakyReluInit :: (BlasM m mx, RandomGen g) => Double -> WeightInitializer g -> g -> m (LeakyReluSt, g)
leakyReluInit sc _ gen = do
  pure (LeakyReluSt sc, gen)
