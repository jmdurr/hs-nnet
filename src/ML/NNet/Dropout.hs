{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module ML.NNet.Dropout where

import Control.Monad.IO.Class
import Data.BlasM
import Data.Proxy
import GHC.TypeLits
import ML.NNet
import System.Random

data DropoutSt g = DropoutSt g Double

data DropoutV mx w h d g = DropoutV g (Matrix mx w h d)

dropoutF :: forall w h d m mx g. (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g) => DropoutSt g -> Matrix mx w h d -> m (Matrix mx w h d, DropoutV mx w h d g)
dropoutF (DropoutSt gen rate) mx = do
  (dmx, gen') <- randomMx (randomR (0.0, 1.0)) gen (Proxy :: Proxy w) (Proxy :: Proxy h) (Proxy :: Proxy d)
  dmx' <- applyFunction dmx (If (IfLt Value (Const rate)) (Const 0) (Const 1))
  r <- mult mx dmx'
  pure (r, DropoutV gen' dmx')

dropoutB :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g) => DropoutSt g -> DropoutV mx w h d g -> Matrix mx w h d -> m (Matrix mx w h d, ())
dropoutB _ (DropoutV _ dv) dz = do
  r <- mult dz dv -- gradients that were dropped out are 0
  pure (r, ())

dropoutAvg :: Monad m => [()] -> m ()
dropoutAvg _ = pure ()

dropoutUpd :: (Monad m, GradientMod m igm mod ()) => DropoutSt g -> () -> igm -> Maybe mod -> m (DropoutSt g, mod)
dropoutUpd st _ igm mod = do
  (_, mod') <- modGradient igm mod ()
  pure (st, mod')

dropout :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g, GradientMod m igm mod ()) => g -> Double -> Layer m mx (DropoutSt g) (DropoutV mx w h d g) () w h d w h d igm mod g
dropout gen rate = Layer dropoutF dropoutB dropoutAvg dropoutUpd (dropoutInit (DropoutSt gen rate))

dropoutInit :: (Monad m, RandomGen g) => (DropoutSt g) -> (g -> (Double, g)) -> g -> m (DropoutSt g, g)
dropoutInit ds _ g = pure (ds, g)
