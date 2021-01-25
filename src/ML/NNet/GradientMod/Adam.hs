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

data AdamSt mx w h d = AdamSt Int (Matrix mx w h d) (Matrix mx w h d)

instance (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => GradientDescentMethod m mx Adam (AdamSt mx w h d) w h d where
  updateWeights (Adam rate b1 b2) st wgt grad = do
    adamD rate b1 b2 st wgt grad

-- (Maybe state -> Matrix mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, state)) ->
-- Ot - (Nebula*mhat) / (vhat + epsilon)

adamD :: (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => Double -> Double -> Double -> Maybe (AdamSt mx w h d) -> Matrix mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, AdamSt mx w h d)
adamD rate b1 b2 Nothing wgt grad = do
  mt <- konst 0.0 -- momentum
  vt <- konst 0.0 -- velocity
  adamD rate b1 b2 (Just (AdamSt 0 mt vt)) wgt grad
adamD rate b1 b2 (Just (AdamSt iter mt vt)) wgt grad = do
  nz <- nearZero
  -- calc momentum
  mt0 <- scale mt b1
  mt2 <- scale grad (1 - b1)
  mt' <- add mt0 mt2

  -- calc velocity
  vt0 <- scale vt b2
  vt1 <- applyFunction grad (Mul (Mul Value Value) (Const (1 - b2)))
  vt' <- add vt0 vt1

  -- mt hat and vt hat
  numer <- applyFunction mt' (Mul (Const (negate rate)) (Div Value (Const (1 - (b1 ** (fromIntegral $ iter + 1))))))
  denom <- applyFunction vt' (Div (Const 1.0) (Add (Sqrt (Div Value (Const (1 - (b2 ** (fromIntegral $ iter + 1)))))) (Const nz)))
  r <- mult numer denom
  wgt' <- add wgt r
  pure (wgt', AdamSt (iter + 1) mt' vt')
