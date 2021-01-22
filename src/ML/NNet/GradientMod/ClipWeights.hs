{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}

module ML.NNet.GradientMod.ClipWeights where

import Data.BlasM
import GHC.TypeLits
import ML.NNet
import ML.NNet.Convolve
import ML.NNet.Deconvolve
import ML.NNet.FullyConnectedSGD
import ML.NNet.LeakyRelu
import ML.NNet.Relu

-- Number of total iterations and underlying algorithm
data ClipWeights gc = ClipWeights Double Double gc

data ClipWeightsSt gm = ClipWeightsSt gm

--     (conf -> Maybe state -> Matrix mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, state))

instance (BlasM m mx, KnownNat w, KnownNat h, KnownNat d, GradientDescentMethod m mx gc gm w h d) => GradientDescentMethod m mx (ClipWeights gc) (ClipWeightsSt gm) w h d where
  updateWeights (ClipWeights mini maxi gc) (Just (ClipWeightsSt gm)) wgt grad = do
    (wgt', gm') <- updateWeights gc (Just gm) wgt grad
    clp <- applyFunction wgt' (Max (Const mini) (Min (Const maxi) Value))
    pure (clp, ClipWeightsSt gm')
  updateWeights (ClipWeights mini maxi gc) Nothing wgt grad = do
    (wgt', gm') <- updateWeights gc Nothing wgt grad
    clp <- applyFunction wgt' (Max (Const mini) (Min (Const maxi) Value))
    pure (clp, ClipWeightsSt gm')
