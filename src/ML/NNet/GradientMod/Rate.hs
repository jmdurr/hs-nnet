{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module ML.NNet.GradientMod.Rate where

import Data.BlasM
import GHC.TypeLits
import ML.NNet

data Rate = Rate Double

-- modGradient :: i -> Maybe a -> b -> m (b, a)
-- class (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => GradientDescentMethod m mx conf state w h d | state mx w h d -> conf where
--     (conf -> Maybe state -> Matrix mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, state))

instance (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => GradientDescentMethod m mx Rate () w h d where
  updateWeights (Rate rate) _ wgt grad = do
    v <- scale grad (- rate)
    m <- add wgt v
    pure (m, ())
  serializeMod _ =
    (pure $ pure $ Just ()
    ,const $ pure $ pure ()
    )
        
