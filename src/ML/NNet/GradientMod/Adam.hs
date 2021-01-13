{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module ML.NNet.GradientMod.Adam where

import Data.BlasM
import GHC.TypeLits
import ML.NNet
import ML.NNet.Convolve
import ML.NNet.Deconvolve
import ML.NNet.FullyConnectedSGD
import ML.NNet.LeakyRelu
import ML.NNet.Relu

data Adam = Adam Double Double Double

data AdamSt mx w h d = AdamSt (Matrix mx w h d) (Matrix mx w h d)

instance (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => GradientMod m Adam (AdamSt mx w h d) (Matrix mx w h d) where
  modGradient (Adam rate b1 b2) st grad = adam rate b1 b2 st grad

instance (Monad m) => GradientMod m Adam () () where
  modGradient _ _ _ = pure ((), ())

instance (Monad m) => GradientMod m Adam () ReluG where
  modGradient _ _ _ = pure (ReluG, ())

instance (Monad m) => GradientMod m Adam () LeakyReluG where
  modGradient _ _ _ = pure (LeakyReluG, ())

data AdamFFN mx i o = AdamFFN
  { aFFw :: AdamSt mx i o 1,
    aFFb :: AdamSt mx 1 o 1
  }

instance (BlasM m mx, KnownNat i, KnownNat o) => GradientMod m Adam (AdamFFN mx i o) (FFNGrad mx i o) where
  modGradient (Adam rate b1 b2) Nothing grad = do
    (wg, rpw) <- adam rate b1 b2 Nothing (ffnWGrad grad)
    (bg, rbp) <- adam rate b1 b2 Nothing (ffnBGrad grad)
    pure (FFNGrad wg bg, AdamFFN rpw rbp)
  modGradient (Adam rate b1 b2) (Just afn) grad = do
    (wg, rpw) <- adam rate b1 b2 (Just $ aFFw afn) (ffnWGrad grad)
    (bg, rpb) <- adam rate b1 b2 (Just $ aFFb afn) (ffnBGrad grad)
    pure (FFNGrad wg bg, AdamFFN rpw rpb)

adam :: (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => Double -> Double -> Double -> Maybe (AdamSt mx w h d) -> Matrix mx w h d -> m (Matrix mx w h d, AdamSt mx w h d)
{-
rmsProp rate _ _ ng = do
  mx <- scale ng rate
  pure (mx, ng)
-}
-- setup a small network and test adam
adam rate b1 b2 Nothing ng = do
  mt <- konst 0.0
  vt <- konst 0.0
  adam rate b1 b2 (Just (AdamSt mt vt)) ng
adam rate b1 b2 (Just (AdamSt mt vt)) ng = do
  mt0 <- scale mt b1
  mt2 <- scale ng (1 - b1)
  mt' <- add mt0 mt2
  vt0 <- scale vt b2
  vt1 <- applyFunction ng (Mul (Mul Value Value) (Const (1 - b2)))
  mth <- applyFunction mt' (Div Value (Const (1 - b1)))
  vth <- applyFunction vt1 (Div Value (Const (1 - b2)))
  gd0 <- applyFunction vth (Div (Const rate) (Add (Sqrt Value) (Const 10e-8)))
  gd1 <- mult gd0 mth
  pure (gd1, AdamSt mt' vt1)
