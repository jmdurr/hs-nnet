{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module ML.NNet.GradientMod.Momentum where

import Data.BlasM
import GHC.TypeLits
import ML.NNet
import ML.NNet.Convolve
import ML.NNet.Deconvolve
import ML.NNet.FullyConnectedSGD
import ML.NNet.LeakyRelu
import ML.NNet.Relu

data Momentum = Momentum Double Double

instance (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => GradientMod m Momentum (Matrix mx w h d) (Matrix mx w h d) where
  modGradient (Momentum rate gain) Nothing grad = do
    g <- scale grad rate
    pure (g, g)
  modGradient (Momentum rate gain) (Just vo) grad = do
    vo1 <- scale vo gain
    vo2 <- scale grad gain
    vo3 <- add vo1 vo2
    pure (vo3, vo3)

instance (Monad m) => GradientMod m Momentum () () where
  modGradient _ _ _ = pure ((), ())

instance (Monad m) => GradientMod m Momentum () ReluG where
  modGradient _ _ _ = pure (ReluG, ())

instance (Monad m) => GradientMod m Momentum () LeakyReluG where
  modGradient _ _ _ = pure (LeakyReluG, ())

data MomentumFFN mx i o = MomentumFFN
  { aFFw :: Matrix mx i o 1,
    aFFb :: Matrix mx 1 o 1
  }

instance (BlasM m mx, KnownNat i, KnownNat o) => GradientMod m Momentum (MomentumFFN mx i o) (FFNGrad mx i o) where
  modGradient (Momentum rate gain) Nothing grad = do
    (_, w) <- modGradient (Momentum rate gain) Nothing (ffnWGrad grad)
    (_, b) <- modGradient (Momentum rate gain) Nothing (ffnBGrad grad)
    pure (FFNGrad w b, MomentumFFN w b)
  modGradient mm (Just (MomentumFFN w b)) (FFNGrad wg bg) = do
    (_, wg') <- modGradient mm (Just w) wg
    (_, bg') <- modGradient mm (Just b) bg
    pure (FFNGrad wg' bg', MomentumFFN wg' bg')
