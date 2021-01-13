{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module ML.NNet.GradientMod.Rate where

import Data.BlasM
import GHC.TypeLits
import ML.NNet
import ML.NNet.Convolve
import ML.NNet.Deconvolve
import ML.NNet.FullyConnectedSGD
import ML.NNet.LeakyRelu
import ML.NNet.Relu

data Rate = Rate Double

-- modGradient :: i -> Maybe a -> b -> m (b, a)

instance (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => GradientMod m Rate () (Matrix mx w h d) where
  modGradient (Rate rate) _ grad = do
    v <- scale grad rate
    pure (v, ())

instance (Monad m) => GradientMod m Rate () () where
  modGradient _ _ _ = pure ((), ())

instance (Monad m) => GradientMod m Rate () ReluG where
  modGradient _ _ _ = pure (ReluG, ())

instance (Monad m) => GradientMod m Rate () LeakyReluG where
  modGradient _ _ _ = pure (LeakyReluG, ())

instance (BlasM m mx, KnownNat i, KnownNat o) => GradientMod m Rate () (FFNGrad mx i o) where
  modGradient (Rate rate) _ grad = do
    wg <- scale (ffnWGrad grad) rate
    bg <- scale (ffnBGrad grad) rate
    pure (FFNGrad wg bg, ())
