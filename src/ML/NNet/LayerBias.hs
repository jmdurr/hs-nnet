{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExplicitForAll #-}
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

data LayerBiasSt mx d mod = LayerBiasSt (Matrix mx 1 1 d) (Maybe mod)

layerBiasForward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => LayerBiasSt mx d mod -> Matrix mx w h d -> m (Matrix mx w h d, ())
layerBiasForward (LayerBiasSt b st) mx = do
  -- add Vector[l] to values in each layer l
  mx' <- addToAllWithDepth mx b
  pure (mx', ())

layerBiasBackward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => LayerBiasSt mx d mod -> () -> Matrix mx w h d -> m (Matrix mx w h d, Matrix mx 1 1 d)
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

layerBiasUpdate :: (Monad m, BlasM m mx, KnownNat d, GradientDescentMethod m mx conf mod 1 1 d) => conf -> LayerBiasSt mx d mod -> Matrix mx 1 1 d -> m (LayerBiasSt mx d mod)
layerBiasUpdate conf (LayerBiasSt bias modst) dbias = do
  (wgt', modst') <- updateWeights conf modst bias dbias
  pure (LayerBiasSt wgt' (Just modst'))

--Layer m mx (ConvolveSt mx fw fh fd di) (ConvolveIn mx wi hi di) (ConvolveG mx fw fh fd di) wi hi di wo ho fd igm mod g
-- (FFN a i o) (FFNIn a i) (FFNGrad a i o)
layerBias :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g, GradientDescentMethod m mx conf igm 1 1 d) => Proxy '(w, h, d) -> Layer m mx (LayerBiasSt mx d igm) () (Matrix mx 1 1 d) w h d w h d conf igm g
layerBias px = Layer layerBiasForward layerBiasBackward layerBiasAverageGrad layerBiasUpdate layerBiasInit

layerBiasInit :: forall m mx g d mod. (Monad m, RandomGen g, BlasM m mx, KnownNat d) => (g -> (Double, g)) -> g -> m (LayerBiasSt mx d mod, g)
layerBiasInit rf g =
  let (b', gen1) = netRandoms rf g (fromIntegral $ natVal (Proxy :: Proxy d))
   in do
        b <- mxFromList b' (Proxy :: Proxy 1) (Proxy :: Proxy 1) (Proxy :: Proxy d)
        pure (LayerBiasSt b Nothing, gen1)
