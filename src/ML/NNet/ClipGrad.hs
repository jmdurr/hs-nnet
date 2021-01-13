{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}

module ML.NNet.ClipGrad where

import Data.BlasM
import Data.Proxy
import GHC.TypeLits
import ML.NNet
import System.Random

data ClipG = ClipG

clipForward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => ClipG -> Matrix mx w h d -> m (Matrix mx w h d, ClipG)
clipForward _ mx = do
  pure (mx, ClipG)

clipBackward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => ClipG -> ClipG -> Matrix mx w h d -> m (Matrix mx w h d, ClipG)
clipBackward _ _ dz = do
  v <- dz `applyFunction` Max (Const (-0.5)) (Min (Const 0.5) Value)
  pure (v, ClipG)

clipAverageGrad :: Monad m => [ClipG] -> m ClipG
clipAverageGrad _ = pure ClipG

clipUpdate :: (Monad m, GradientMod m igm a ClipG) => ClipG -> ClipG -> igm -> Maybe a -> m (ClipG, a)
clipUpdate _ _ igm a = do
  (_, mod') <- modGradient igm a ClipG
  pure (ClipG, mod')

clip :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g, GradientMod m igm gmod ClipG) => Proxy '(w, h, d) -> Layer m mx ClipG ClipG ClipG w h d w h d igm gmod g
clip px = Layer clipForward clipBackward clipAverageGrad clipUpdate clipInit

clipInit :: (Monad m, RandomGen g) => (g -> (Double, g)) -> g -> m (ClipG, g)
clipInit _ g = do
  pure (ClipG, g)
