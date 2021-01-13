{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module ML.NNet.GradientMod.RMSProp where

import Data.BlasM
import GHC.TypeLits
import ML.NNet
import ML.NNet.Convolve
import ML.NNet.Deconvolve
import ML.NNet.FullyConnectedSGD
import ML.NNet.LeakyRelu
import ML.NNet.Relu

data RMSProp = RMSProp Double Double

instance (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => GradientMod m RMSProp (Matrix mx w h d) (Matrix mx w h d) where
  modGradient (RMSProp rate beta) st grad = rmsProp rate beta st grad

instance (Monad m) => GradientMod m RMSProp () () where
  modGradient _ _ _ = pure ((), ())

instance (Monad m) => GradientMod m RMSProp () ReluG where
  modGradient _ _ _ = pure (ReluG, ())

instance (Monad m) => GradientMod m RMSProp () LeakyReluG where
  modGradient _ _ _ = pure (LeakyReluG, ())

instance (BlasM m mx, KnownNat i, KnownNat o) => GradientMod m RMSProp (FFNGrad mx i o) (FFNGrad mx i o) where
  modGradient (RMSProp rate beta) st grad = do
    (wg, rpw) <- rmsProp rate beta (ffnWGrad <$> st) (ffnWGrad grad)
    (bg, rpb) <- rmsProp rate beta (ffnBGrad <$> st) (ffnBGrad grad)
    pure (FFNGrad wg bg, FFNGrad rpw rpb)

rmsProp :: (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => Double -> Double -> Maybe (Matrix mx w h d) -> Matrix mx w h d -> m (Matrix mx w h d, Matrix mx w h d)
{-
rmsProp rate _ _ ng = do
  mx <- scale ng rate
  pure (mx, ng)
-}

rmsProp rate beta Nothing ng = do
  eg <- applyFunction ng (Mul (Mul Value Value) (Const (1.0 - beta)))
  d1 <- applyFunction eg (Div (Const rate) (Add (Sqrt Value) (Const 1e-7)))
  d2 <- mult d1 ng
  pure (d2, eg)
rmsProp rate beta (Just st) ng = do
  st' <- scale st beta
  eg <- applyFunction ng (Mul (Mul Value Value) (Const (1.0 - beta)))
  neg <- add st' eg
  d1 <- applyFunction neg (Div (Const rate) (Add (Sqrt Value) (Const 1e-7)))
  d2 <- mult d1 ng
  pure (d2, neg)
