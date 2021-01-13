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
import System.Random

data LeakyReluSt = LeakyReluSt

type LeakyReluIn mx w h d = Matrix mx w h d

data LeakyReluG = LeakyReluG

leakyReluForward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => LeakyReluSt -> Matrix mx w h d -> m (Matrix mx w h d, LeakyReluIn mx w h d)
leakyReluForward _ mx = do
  mx' <- mx `applyFunction` (If (IfLt Value (Const 0.0)) (Mul (Const 0.2) Value) Value) -- 0 if input is < 0 else input
  pure (mx', mx)

leakyReluBackward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => LeakyReluSt -> LeakyReluIn mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, LeakyReluG)
leakyReluBackward _ inp mx = do
  -- if input is < 0 then 0 else 1
  -- max 0 Value / Value
  -- input is stored in LeakyRelu
  -- v <- applyFunction inp (Max (Const 0.0) Value `Div` Max (Const 0.00000000000001) Value) -- if input is < 0 make it 0, divide by original value to make 1 but if it is 0 we need it to be 1...?
  v <- applyFunction inp (If (IfGt Value (Const 0.0)) (Const 1.0) (Const 0.02)) -- run this on input to generate 0 or 1
  v' <- mult mx v
  -- multiply 1 for 1 with mx
  pure (v', LeakyReluG)

leakyReluG :: Monad m => [LeakyReluG] -> m LeakyReluG
leakyReluG _ = pure LeakyReluG

leakyReluU :: (Monad m, GradientMod m igm mod LeakyReluG) => LeakyReluSt -> LeakyReluG -> igm -> Maybe mod -> m (LeakyReluSt, mod)
leakyReluU _ _ igm mod = do
  (_, mod') <- modGradient igm mod LeakyReluG
  pure (LeakyReluSt, mod')

leakyRelu :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g, GradientMod m igm mod LeakyReluG) => Proxy w -> Proxy h -> Proxy d -> Layer m mx LeakyReluSt (LeakyReluIn mx w h d) LeakyReluG w h d w h d igm mod g
leakyRelu w h d = Layer leakyReluForward leakyReluBackward leakyReluG leakyReluU leakyReluInit

leakyReluInit :: (BlasM m mx, RandomGen g) => (g -> (Double, g)) -> g -> m (LeakyReluSt, g)
leakyReluInit _ gen = do
  pure (LeakyReluSt, gen)
