{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module ML.NNet.Relu where

import Data.BlasM
import Data.Proxy
import Debug.Trace
import GHC.TypeLits
import ML.NNet
import ML.NNet.Init.RandomFun
import System.Random

data ReluSt = ReluSt

type ReluIn mx w h d = Matrix mx w h d

data ReluG = ReluG

reluForward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => ReluSt -> Matrix mx w h d -> m (Matrix mx w h d, ReluIn mx w h d)
reluForward _ mx = do
  mx' <- mx `applyFunction` Max (Const 0.0) Value -- 0 if input is < 0 else input
  pure (mx', mx)

reluBackward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => ReluSt -> ReluIn mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, ReluG)
reluBackward _ oldIn mx = do
  -- if input is < 0 then 0 else 1
  -- max 0 Value / Value
  -- input is stored in Relu
  -- v <- applyFunction inp (Max (Const 0.0) Value `Div` Max (Const 0.00000000000001) Value) -- if input is < 0 make it 0, divide by original value to make 1 but if it is 0 we need it to be 1...?
  v' <- applyFunction oldIn (If (IfLt Value (Const 0.0)) (Const 0.0) (Const 1.0))
  dy <- mult v' mx
  pure (dy, ReluG)

reluAvg :: Monad m => [ReluG] -> m ReluG
reluAvg _ = pure ReluG

reluU :: (Monad m) => conf -> ReluSt -> ReluG -> m ReluSt
reluU _ st _ = pure st

relu :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g) => Proxy w -> Proxy h -> Proxy d -> Layer m mx ReluSt (ReluIn mx w h d) ReluG w h d w h d conf mod g
relu w h d = Layer reluForward reluBackward reluAvg reluU reluInit

reluInit :: (RandomGen g, Monad m) => WeightInitializer g -> g -> m (ReluSt, g)
reluInit _ gen = pure (ReluSt, gen)
