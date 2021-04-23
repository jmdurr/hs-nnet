{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
module ML.NNet.GradientMod.Momentum where

import Data.BlasM
import GHC.TypeLits
import ML.NNet
import Data.Proxy

data Momentum = Momentum Double Double

type MomentumSt mx w h d = Matrix mx w h d

--instance (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => GradientDescentMethod m mx (RMSProp) (RMSPropSt mx w h d) w h d where

instance (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => GradientDescentMethod m mx Momentum (MomentumSt mx w h d) w h d where
  updateWeights (Momentum rate gain) Nothing wgt grad = do
    vdw <- scale grad (1 - gain)
    vdws <- scale vdw (- rate)
    wgt' <- add wgt vdws
    pure (wgt', vdw)
  updateWeights (Momentum rate gain) (Just vo) wgt grad = do
    vo1 <- scale vo gain
    vo2 <- scale grad (1 - gain)
    vo3 <- add vo1 vo2
    vo4 <- scale vo3 (- rate)
    wgt' <- add wgt vo4
    pure (wgt', vo3)
  serializeMod _ =
    ( maybeDeserializeMx (Proxy :: Proxy '(w,h,d))
    , maybeSerializeMx
    )
