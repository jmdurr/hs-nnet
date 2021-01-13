{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}

module ML.NNet.GradientMod.Demon where

import Data.BlasM
import GHC.TypeLits
import ML.NNet
import ML.NNet.Convolve
import ML.NNet.Deconvolve
import ML.NNet.FullyConnectedSGD
import ML.NNet.LeakyRelu
import ML.NNet.Relu

-- Number of total iterations and underlying algorithm
data Demon a = Demon Int Double (Double -> a)

data DemonSt b = DemonSt Int b

demonCalc :: Double -> Int -> Int -> Double
demonCalc minit iter tIter =
  let rs = ((1.0 - (fromIntegral iter / fromIntegral tIter)) * minit) / (1.0 - minit)
   in rs / (1.0 + rs)

--   modGradient :: i -> Maybe a -> b -> m (b, a)
dmod :: (Monad m, GradientMod m a b g) => Demon a -> Maybe (DemonSt b) -> g -> m (g, DemonSt b)
dmod (Demon titer minit fa) Nothing grad = do
  let nm = demonCalc minit 1 titer
  (grad', ast') <- ML.NNet.modGradient (fa nm) Nothing grad
  pure (grad', DemonSt 1 ast')
dmod (Demon titer minit fa) (Just (DemonSt citer ast)) grad = do
  let nm = demonCalc minit citer titer
  (grad', ast') <- ML.NNet.modGradient (fa nm) (Just ast) grad
  pure (grad', DemonSt (citer + 1) ast')

instance (BlasM m mx, KnownNat w, KnownNat h, KnownNat d, GradientMod m a b (Matrix mx w h d)) => GradientMod m (Demon a) (DemonSt b) (Matrix mx w h d) where
  modGradient d dst grad = dmod d dst grad

instance (Monad m, GradientMod m a () ()) => GradientMod m (Demon a) () () where
  modGradient _ _ _ = pure ((), ())

instance (Monad m, GradientMod m a () ReluG) => GradientMod m (Demon a) () ReluG where
  modGradient _ _ _ = pure (ReluG, ())

instance (Monad m, GradientMod m a () LeakyReluG) => GradientMod m (Demon a) () LeakyReluG where
  modGradient _ _ _ = pure (LeakyReluG, ())

instance (BlasM m mx, KnownNat i, KnownNat o, GradientMod m a b (FFNGrad mx i o)) => GradientMod m (Demon a) (DemonSt b) (FFNGrad mx i o) where
  modGradient d dst grad = dmod d dst grad
