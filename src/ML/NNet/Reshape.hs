{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module ML.NNet.Reshape where

import Data.BlasM
import Data.Serialize
import GHC.TypeLits
import ML.NNet
import ML.NNet.Init.RandomFun
import System.Random

reshapeSerialize :: Monad m => conf -> (Get (m ()), () -> m Put)
reshapeSerialize _ =
  ( pure $ pure (()),
    const (pure (pure ()))
  )

reshapeBackward ::
  forall w h d w2 h2 d2 m mx.
  (KnownNat w, KnownNat h, KnownNat d, KnownNat w2, KnownNat h2, KnownNat d2, BlasM m mx, KnownNat (w GHC.TypeLits.* h), KnownNat (w2 GHC.TypeLits.* h2), (w GHC.TypeLits.* h GHC.TypeLits.* d) ~ (w2 GHC.TypeLits.* h2 GHC.TypeLits.* d2)) =>
  () ->
  () ->
  Matrix mx w h d ->
  m (Matrix mx w2 h2 d2, ())
reshapeBackward _ _ mx = (,()) <$> reshapeM mx

reshapeForward ::
  (KnownNat w, KnownNat h, KnownNat d, KnownNat w2, KnownNat h2, KnownNat d2, BlasM m mx, KnownNat (w GHC.TypeLits.* h), KnownNat (w2 GHC.TypeLits.* h2), (w GHC.TypeLits.* h GHC.TypeLits.* d) ~ (w2 GHC.TypeLits.* h2 GHC.TypeLits.* d2)) =>
  () ->
  Matrix mx w h d ->
  m (Matrix mx w2 h2 d2, ())
reshapeForward _ mx =
  (,()) <$> reshapeM mx

reshapeAvg :: Monad m => (() -> () -> m (), () -> Int -> m ())
reshapeAvg = (const (const (pure ())), const (const (pure ())))

reshapeUpd ::
  (Monad m) =>
  conf ->
  () ->
  () ->
  m ()
reshapeUpd _ _ _ = pure ()

reshape ::
  (KnownNat w, KnownNat h, KnownNat d, KnownNat w2, KnownNat h2, KnownNat d2, KnownNat (w GHC.TypeLits.* h), KnownNat (w2 GHC.TypeLits.* h2), (w GHC.TypeLits.* h GHC.TypeLits.* d) ~ (w2 GHC.TypeLits.* h2 GHC.TypeLits.* d2), BlasM m mx, RandomGen g) =>
  Layer m mx () () () w h d w2 h2 d2 gst mod g
reshape = Layer reshapeForward reshapeBackward reshapeAvg reshapeUpd reshapeInit reshapeSerialize

reshapeInit :: (Monad m, RandomGen g) => WeightInitializer g -> g -> m ((), g)
reshapeInit _ g = pure ((), g)
