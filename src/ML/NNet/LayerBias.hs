{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module ML.NNet.LayerBias where

-- one bias per layer as used by convolve and deconvolve

import Data.BlasM
import Data.Proxy
import GHC.TypeLits
import ML.NNet
import System.Random

layerBiasForward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => Matrix mx 1 1 d -> Matrix mx w h d -> m (Matrix mx w h d, ())
layerBiasForward b mx = do
  -- add Vector[l] to values in each layer l
  mx' <- addToAllWithDepth mx b
  pure (mx', ())

layerBiasBackward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => Matrix mx 1 1 d -> () -> Matrix mx w h d -> m (Matrix mx w h d, Matrix mx 1 1 d)
layerBiasBackward bias _ dz = do
  b' <- sumLayers dz
  pure (dz, b')

layerBiasAverageGrad :: (BlasM m mx, KnownNat d, Monad m) => [Matrix mx 1 1 d] -> m (Matrix mx 1 1 d)
layerBiasAverageGrad [] = error "Cannot average empty gradient"
layerBiasAverageGrad (g : gs) = go g gs
  where
    go g' [] = do
      wg <- applyFunction g' (Div Value (Const (fromIntegral $ 1 + length gs)))
      pure $ wg
    go g' (ng : gs') = do
      vw <- add g' ng
      go vw gs'

layerBiasUpdate :: (Monad m, BlasM m mx, KnownNat d, GradientMod m igm a (Matrix mx 1 1 d)) => Matrix mx 1 1 d -> Matrix mx 1 1 d -> igm -> Maybe a -> m (Matrix mx 1 1 d, a)
layerBiasUpdate bias dbias igm a = do
  (db', mod') <- modGradient igm a dbias
  v <- subtractM bias db'
  pure (v, mod')

--Layer m mx (ConvolveSt mx fw fh fd di) (ConvolveIn mx wi hi di) (ConvolveG mx fw fh fd di) wi hi di wo ho fd igm mod g
-- (FFN a i o) (FFNIn a i) (FFNGrad a i o)
layerBias :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g, GradientMod m igm gmod (Matrix mx 1 1 d)) => Proxy '(w, h, d) -> Layer m mx (Matrix mx 1 1 d) () (Matrix mx 1 1 d) w h d w h d igm gmod g
layerBias px = Layer layerBiasForward layerBiasBackward layerBiasAverageGrad layerBiasUpdate layerBiasInit

layerBiasInit :: (Monad m, RandomGen g, BlasM m mx, KnownNat d) => (g -> (Double, g)) -> g -> m (Matrix mx 1 1 d, g)
layerBiasInit _ g = do
  b <- konst 0.0
  pure (b, g)
