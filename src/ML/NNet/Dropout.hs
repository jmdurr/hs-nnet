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
import ML.NNet.Init.RandomFun
import System.Random

data DropoutSt mx w h d g = DropoutSt g Double (Matrix mx w h d)

dropoutF :: forall w h d m mx g. (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g) => DropoutSt mx w h d g -> Matrix mx w h d -> m (Matrix mx w h d, ())
dropoutF (DropoutSt gen rate dmx) mx = do
  r <- mult mx dmx
  pure (r, ())

dropoutB :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g) => DropoutSt mx w h d g -> () -> Matrix mx w h d -> m (Matrix mx w h d, ())
dropoutB (DropoutSt gen rate dmx) _ dz = do
  r <- mult dz dmx -- gradients that were dropped out are 0
  pure (r, ())

dropoutAvg :: Monad m => [()] -> m ()
dropoutAvg _ = pure ()

-- (gconf -> lst -> gd -> m lst) ->
dropoutUpd :: forall m mx g w h d conf. (BlasM m mx, RandomGen g, KnownNat w, KnownNat h, KnownNat d) => conf -> DropoutSt mx w h d g -> () -> m (DropoutSt mx w h d g)
dropoutUpd _ (DropoutSt gen rate dmx) _ = do
  (dmx1, gen') <- randomMx (\_ _ -> randomR (0.0, 1.0)) gen (Proxy :: Proxy w) (Proxy :: Proxy h) (Proxy :: Proxy d)
  dmx2 <- applyFunction dmx1 (If (IfLt Value (Const rate)) (Const 0) (Const 1))
  pure (DropoutSt gen' rate dmx2)

dropout :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g) => g -> Double -> Layer m mx (DropoutSt mx w h d g) () () w h d w h d gconf mod g
dropout gen rate = Layer dropoutF dropoutB dropoutAvg dropoutUpd (dropoutInit gen rate)

dropoutInit :: forall w h d m mx g. (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g) => g -> Double -> WeightInitializer g -> g -> m (DropoutSt mx w h d g, g)
dropoutInit gen rate _ g = do
  (dmx, gen') <- randomMx (\_ _ -> randomR (0.0, 1.0)) gen (Proxy :: Proxy w) (Proxy :: Proxy h) (Proxy :: Proxy d)
  dmx' <- applyFunction dmx (If (IfLt Value (Const rate)) (Const 0) (Const 1))
  pure (DropoutSt gen' rate dmx', g)
