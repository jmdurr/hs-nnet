{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

{-
Deconvolution is a convolution with dialated input
-}
module ML.NNet.Deconvolve where

import Data.BlasM
import Data.Proxy
import Debug.Trace
import GHC.TypeLits
import ML.NNet
import ML.NNet.LayerBias
import System.Random

type DeconvolveSt mx fw fh fd id = Matrix mx fw fh (id GHC.TypeLits.* fd)

type DeconvolveIn mx iw ih id = Matrix mx iw ih id

type DeconvolveG mx fw fh fd id = Matrix mx fw fh (id GHC.TypeLits.* fd)

deconForward ::
  forall m mx wi hi di fw fh fd wo ho pyt pyb pxl pxr sy sx.
  ( BlasM m mx,
    KnownNat wi,
    KnownNat hi,
    KnownNat di,
    KnownNat fw,
    KnownNat fh,
    KnownNat fd,
    KnownNat wo,
    KnownNat ho,
    KnownNat pyt,
    KnownNat pyb,
    KnownNat pxl,
    KnownNat pxr,
    KnownNat sy,
    KnownNat sx,
    KnownNat (ho - 1),
    KnownNat (wo - 1),
    KnownNat (hi - 1),
    KnownNat (wi - 1),
    KnownNat (sx - 1),
    KnownNat (sy - 1),
    (((((hi + ((hi - 1) GHC.TypeLits.* (sy - 1))) + pyt) + pyb) - fh) + 1) ~ ho,
    (((((wi + ((wi - 1) GHC.TypeLits.* (sx - 1))) + pxl) + pxr) - fw) + 1) ~ wo,
    Mod (di GHC.TypeLits.* fd) fd ~ 0,
    Div (di GHC.TypeLits.* fd) fd ~ di,
    KnownNat (di GHC.TypeLits.* fd)
  ) =>
  Proxy '(sx, sy) ->
  Proxy '(pxl, pxr, pyt, pyb) ->
  DeconvolveSt mx fw fh fd di ->
  Matrix mx wi hi di ->
  m (Matrix mx wo ho fd, DeconvolveIn mx wi hi di)
deconForward ps pp wgt inmx = do
  -- filter is smaller than input
  r <- Data.BlasM.convolve inmx wgt (Proxy :: Proxy fd) (Proxy :: Proxy 1, Proxy :: Proxy 1) (Proxy :: Proxy pxl, Proxy :: Proxy pxr, Proxy :: Proxy pyt, Proxy :: Proxy pyb) (Proxy :: Proxy (sx - 1), Proxy :: Proxy (sy - 1)) (Proxy :: Proxy 0, Proxy :: Proxy 0) False
  -- r' <- add r (cFilterBiases con) -- each filter layer has 1 bias so just need to add it to every output at each depth
  -- need to add each bias in each layer to each layer of output...
  --r' <- addToAllWithDepth r (cFilterBiases con)
  pure (r, inmx)

deconBackward ::
  forall m mx wi hi di fw fh fd sx sy pxl pxr pyt pyb wo ho.
  ( BlasM m mx,
    KnownNat wi,
    KnownNat hi,
    KnownNat di,
    KnownNat fw,
    KnownNat (fw - 1),
    KnownNat fh,
    KnownNat (fh - 1),
    KnownNat fd,
    KnownNat sx,
    KnownNat sy,
    KnownNat (sy - 1),
    KnownNat (sx - 1),
    KnownNat pxl,
    KnownNat pxr,
    KnownNat pyt,
    KnownNat pyb,
    KnownNat wo,
    KnownNat ho,
    KnownNat (ho - 1),
    KnownNat (wo - 1),
    KnownNat (hi - 1),
    KnownNat (wi - 1),
    KnownNat ((fw - 1) - pxl),
    KnownNat ((fw - 1) - pxr),
    KnownNat ((fh - 1) - pyt),
    KnownNat ((fh - 1) - pyb),
    (((((hi + ((hi - 1) GHC.TypeLits.* (sy - 1))) + pyt) + pyb) - fh) + 1) ~ ho,
    (((((wi + ((wi - 1) GHC.TypeLits.* (sx - 1))) + pxl) + pxr) - fw) + 1) ~ wo,
    Mod (((wo + ((fw - 1) - pxl)) + ((fw - 1) - pxr)) - fw) sx ~ 0,
    Mod (((ho + ((fh - 1) - pyt)) + ((fh - 1) - pyb)) - fh) sy ~ 0,
    ((((ho + (fh - 1)) + (fh - 1)) - fh) + sy) ~ ((sy GHC.TypeLits.* hi) + pyt + pyb),
    ((((wo + (fw - 1)) + (fw - 1)) - fw) + sx) ~ ((sx GHC.TypeLits.* wi) + pxl + pxr),
    (((((hi + ((hi - 1) GHC.TypeLits.* (sy - 1))) + pyt) + pyb) - ho) + 1) ~ fh,
    (((((wi + ((wi - 1) GHC.TypeLits.* (sx - 1))) + pxl) + pxr) - wo) + 1) ~ fw,
    ((((wo + ((fw - 1) - pxl)) + ((fw - 1) - pxr)) - fw) + sx) ~ (sx GHC.TypeLits.* wi),
    ((((ho + ((fh - 1) - pyt)) + ((fh - 1) - pyb)) - fh) + sy) ~ (sy GHC.TypeLits.* hi),
    Mod (di GHC.TypeLits.* fd) fd ~ 0,
    Div (di GHC.TypeLits.* fd) fd ~ di,
    KnownNat (di GHC.TypeLits.* fd)
  ) =>
  Proxy '(sx, sy) ->
  Proxy '(pxl, pxr, pyt, pyb) ->
  DeconvolveSt mx fw fh fd di ->
  DeconvolveIn mx wi hi di ->
  Matrix mx wo ho fd ->
  m (Matrix mx wi hi di, DeconvolveG mx fw fh fd di)
deconBackward ps pp wgt oldIn dy = do
  -- for deconvolution might need to take dialation into account for both forward and backward prop
  dLdX <- Data.BlasM.convolveLayersDy dy wgt (Proxy :: Proxy sx, Proxy :: Proxy sy) (Proxy :: Proxy (fw - 1 - pxl), Proxy :: Proxy (fw - 1 - pxr), Proxy :: Proxy (fh - 1 - pyt), Proxy :: Proxy (fh - 1 - pyb)) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0) True
  dLdW <- Data.BlasM.convolveLayers oldIn dy (Proxy :: Proxy 1, Proxy :: Proxy 1) (Proxy :: Proxy pxl, Proxy :: Proxy pxr, Proxy :: Proxy pyt, Proxy :: Proxy pyb) (Proxy :: Proxy (sx - 1), Proxy :: Proxy (sy - 1)) (Proxy :: Proxy 0, Proxy :: Proxy 0) False
  pure (dLdX, dLdW)

deconAvg :: (BlasM m mx, KnownNat fw, KnownNat fh, KnownNat fd, KnownNat di, KnownNat (di GHC.TypeLits.* fd)) => Proxy '(fw, fh, fd, di) -> [DeconvolveG mx fw fh fd di] -> m (DeconvolveG mx fw fh fd di)
deconAvg _ = avgMxs

deconUpd :: (BlasM m mx, KnownNat fw, KnownNat fh, KnownNat fd, KnownNat di, KnownNat (di GHC.TypeLits.* fd), GradientMod m igm mod (DeconvolveG mx fw fh fd di)) => Proxy '(fw, fh, fd, di) -> DeconvolveSt mx fw fh fd di -> DeconvolveG mx fw fh fd di -> igm -> Maybe mod -> m (DeconvolveSt mx fw fh fd di, mod)
deconUpd _ wgt dw igm mod = do
  (dw', mod') <- modGradient igm mod dw
  dwOut <- subtractM wgt dw'
  pure (dwOut, mod')

deconInit ::
  forall g m mx fw fh fd di.
  ( RandomGen g,
    BlasM m mx,
    KnownNat fd,
    KnownNat fw,
    KnownNat fh,
    KnownNat di,
    Mod (di GHC.TypeLits.* fd) fd ~ 0,
    Div (di GHC.TypeLits.* fd) fd ~ di,
    KnownNat (di GHC.TypeLits.* fd)
  ) =>
  Proxy '(fw, fh, fd, di) ->
  (g -> (Double, g)) ->
  g ->
  m (DeconvolveSt mx fw fh fd di, g)
deconInit _ rf gen =
  let (filt, gen1) = netRandoms rf gen (fromIntegral $ natVal (Proxy :: Proxy fw) * natVal (Proxy :: Proxy fh) * natVal (Proxy :: Proxy (di GHC.TypeLits.* fd)))
   in do
        fmx <- mxFromList filt (Proxy :: Proxy fw) (Proxy :: Proxy fh) (Proxy :: Proxy (di GHC.TypeLits.* fd))
        pure $ (fmx, gen1)

deconvolve ::
  forall m mx wi hi di wo ho g fw fh fd sx sy pxl pxr pyt pyb amod bmod igm.
  ( BlasM m mx,
    KnownNat wi,
    KnownNat hi,
    KnownNat di,
    KnownNat wo,
    KnownNat ho,
    KnownNat fd,
    RandomGen g,
    KnownNat fw,
    KnownNat fh,
    KnownNat sx,
    KnownNat sy,
    KnownNat pxl,
    KnownNat pxr,
    KnownNat pyt,
    KnownNat pyb,
    KnownNat (fw - 1),
    KnownNat (fh - 1),
    KnownNat (sy - 1),
    KnownNat (sx - 1),
    KnownNat (ho - 1),
    KnownNat (wo - 1),
    KnownNat (hi - 1),
    KnownNat (wi - 1),
    KnownNat ((fw - 1) - pxl),
    KnownNat ((fw - 1) - pxr),
    KnownNat ((fh - 1) - pyt),
    KnownNat ((fh - 1) - pyb),
    (((((hi + ((hi - 1) GHC.TypeLits.* (sy - 1))) + pyt) + pyb) - fh) + 1) ~ ho,
    (((((wi + ((wi - 1) GHC.TypeLits.* (sx - 1))) + pxl) + pxr) - fw) + 1) ~ wo,
    (((((hi + ((hi - 1) GHC.TypeLits.* (sy - 1))) + pyt) + pyb) - ho) + 1) ~ fh,
    (((((wi + ((wi - 1) GHC.TypeLits.* (sx - 1))) + pxl) + pxr) - wo) + 1) ~ fw,
    Mod (((wo + ((fw - 1) - pxl)) + ((fw - 1) - pxr)) - fw) sx ~ 0,
    Mod (((ho + ((fh - 1) - pyt)) + ((fh - 1) - pyb)) - fh) sy ~ 0,
    ((((ho + (fh - 1)) + (fh - 1)) - fh) + sy) ~ ((sy GHC.TypeLits.* hi) + pyt + pyb),
    ((((wo + (fw - 1)) + (fw - 1)) - fw) + sx) ~ ((sx GHC.TypeLits.* wi) + pxl + pxr),
    ((((wo + ((fw - 1) - pxl)) + ((fw - 1) - pxr)) - fw) + sx) ~ (sx GHC.TypeLits.* wi),
    ((((ho + ((fh - 1) - pyt)) + ((fh - 1) - pyb)) - fh) + sy) ~ (sy GHC.TypeLits.* hi),
    Mod (di GHC.TypeLits.* fd) fd ~ 0,
    Div (di GHC.TypeLits.* fd) fd ~ di,
    KnownNat (di GHC.TypeLits.* fd),
    GradientMod m igm (amod, bmod) (DeconvolveG mx fw fh fd di, Matrix mx 1 1 fd),
    GradientMod m igm amod (DeconvolveG mx fw fh fd di),
    GradientMod m igm bmod (Matrix mx 1 1 fd)
  ) =>
  -- | filter size, dpo is the expected output size the real filter is (fd = -dpo + id + 1)
  Proxy '(fw, fh, fd) ->
  -- | Input dialation size
  Proxy '(sx, sy) ->
  -- | Padding size
  Proxy '(pxl, pxr, pyt, pyb) ->
  Layer m mx (DeconvolveSt mx fw fh fd di, Matrix mx 1 1 fd) (DeconvolveIn mx wi hi di, ()) (DeconvolveG mx fw fh fd di, Matrix mx 1 1 fd) wi hi di wo ho fd igm (amod, bmod) g
deconvolve pxf pxs pxp =
  let px = Proxy :: Proxy '(fw, fh, fd, di)
   in (Layer (deconForward pxs pxp) (deconBackward pxs pxp) (deconAvg px) (deconUpd px) (deconInit px)) `connect` layerBias (Proxy :: Proxy '(wo, ho, fd))
