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

type MomentumSt mx w h d = Matrix mx w h d

--instance (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => GradientDescentMethod m mx (RMSProp) (RMSPropSt mx w h d) w h d where

instance (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => GradientDescentMethod m mx Momentum (MomentumSt mx w h d) w h d where
  updateWeights (Momentum rate gain) Nothing wgt grad = do
    vo2 <- scale grad (- rate)
    wgt' <- add wgt vo2
    pure (wgt', vo2)
  updateWeights (Momentum rate gain) (Just vo) wgt grad = do
    vo1 <- scale vo gain
    vo2 <- scale grad (- rate)
    vo3 <- add vo1 vo2
    wgt' <- add wgt vo3
    pure (wgt', vo3)
