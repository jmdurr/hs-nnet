{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module ML.NNet.Dropout where

import Data.BlasM
import Data.Proxy
import Data.Serialize
import GHC.TypeLits
import ML.NNet
import ML.NNet.Init.RandomFun
import System.Random

data DropoutSt mx w h d g = DropoutSt g Double (Matrix mx w h d)

dropoutF :: forall w h d m mx g. (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g) => DropoutSt mx w h d g -> Matrix mx w h d -> m (Matrix mx w h d, ())
dropoutF (DropoutSt _ _ dmx) mx = do
  r <- mult mx dmx
  pure (r, ())

dropoutB :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g) => DropoutSt mx w h d g -> () -> Matrix mx w h d -> m (Matrix mx w h d, ())
dropoutB (DropoutSt _ _ dmx) _ dz = do
  r <- mult dz dmx
  pure (r, ())

dropoutAvg :: Monad m => (() -> () -> m (), () -> Int -> m ())
dropoutAvg = (const (const (pure ())), const (const (pure ())))

-- (gconf -> lst -> gd -> m lst) ->
dropoutUpd :: forall m mx g w h d conf. (BlasM m mx, RandomGen g, KnownNat w, KnownNat h, KnownNat d) => conf -> DropoutSt mx w h d g -> () -> m (DropoutSt mx w h d g)
dropoutUpd _ (DropoutSt gen rate _) _ = do
  (dmx1, gen') <- randomMx (\_ _ -> randomR (0.0, 1.0)) gen (Proxy :: Proxy w) (Proxy :: Proxy h) (Proxy :: Proxy d) 0 0
  dmx2 <- applyFunction dmx1 (If (IfLt Value (Const rate)) (Const 0) (Const (1.0 / (1.0 - rate))))
  pure (DropoutSt gen' rate dmx2)

dropout :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g) => g -> Double -> Layer m mx (DropoutSt mx w h d g) () () w h d w h d gconf mod g
dropout gen rate = Layer dropoutF dropoutB dropoutAvg dropoutUpd (dropoutInit gen rate) (dropoutSerialize gen)

dropoutSerialize :: forall m mx w h d g conf. (BlasM m mx, KnownNat w, KnownNat h, KnownNat d, RandomGen g) => g -> conf -> (Get (m (DropoutSt mx w h d g)), DropoutSt mx w h d g -> m Put)
dropoutSerialize g _ =
  ( do
      r <- getFloat64be
      mx <- deserializeMx (Proxy :: Proxy '(w, h, d))
      pure $ DropoutSt g r <$> mx,
    \(DropoutSt _ d mx) ->
      do
        m <- serializeMx mx
        pure (putFloat64be d >> m)
  )

dropoutInit :: forall w h d m mx g. (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g) => g -> Double -> WeightInitializer g -> g -> m (DropoutSt mx w h d g, g)
dropoutInit gen rate _ g = do
  (dmx, gen') <- randomMx (\_ _ -> randomR (0.0, 1.0)) gen (Proxy :: Proxy w) (Proxy :: Proxy h) (Proxy :: Proxy d) 0 0
  dmx' <- applyFunction dmx (If (IfLt Value (Const rate)) (Const 0) (Const (1.0 / (1.0 - rate))))
  pure (DropoutSt gen' rate dmx', g)
