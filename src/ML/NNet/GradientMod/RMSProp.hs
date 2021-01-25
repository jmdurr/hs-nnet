{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

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

type RMSPropSt mx w h d = Matrix mx w h d

--     (conf -> Maybe state -> Matrix mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, state))

instance (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => GradientDescentMethod m mx (RMSProp) (RMSPropSt mx w h d) w h d where
  updateWeights (RMSProp rate beta) st wgt grad = do
    nz <- nearZero
    v2 <- applyFunction grad (Mul (Const (1 - beta)) (Mul Value Value))
    vt <- case st of
      Nothing -> pure v2
      Just v1 -> do
        v1' <- applyFunction v1 (Mul Value (Const beta))
        add v1' v2
    dwt1 <- applyFunction vt (Div (Const (- rate)) (Sqrt (Add Value (Const nz))))
    dwt <- mult dwt1 grad
    wt <- add wgt dwt
    pure (wt, vt)
