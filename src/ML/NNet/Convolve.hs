{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module ML.NNet.Convolve where

import Control.Monad.IO.Class
import Data.BlasM
import Data.Proxy
import Debug.Trace
import GHC.TypeLits
import ML.NNet
import ML.NNet.LayerBias
import System.Random

{-
A filter must always have the same # of layers as input channels
Therefore multiple filters would have f * l layers and each filter is applied
to all channels

This results in a single feature map per filter or 1 output layer per filter

-}

type ConvolveSt mx fw fh fd id = Matrix mx fw fh (id GHC.TypeLits.* fd)

type ConvolveIn mx iw ih id = Matrix mx iw ih id

type ConvolveG mx fw fh fd id = Matrix mx fw fh (id GHC.TypeLits.* fd)

conForward ::
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
    ((hi + pyt + pyb - fh + sy) ~ (sy GHC.TypeLits.* ho)),
    ((wi + pxl + pxr - fw + sx) ~ (sx GHC.TypeLits.* wo)),
    Mod (wi + pxl + pxr - fw) sx ~ 0,
    Mod (hi + pyt + pyb - fh) sy ~ 0,
    KnownNat (di GHC.TypeLits.* fd)
  ) =>
  Proxy '(sx, sy) ->
  Proxy '(pxl, pxr, pyt, pyb) ->
  ConvolveSt mx fw fh fd di ->
  Matrix mx wi hi di ->
  m (Matrix mx wo ho fd, ConvolveIn mx wi hi di)
conForward ps pp flt inmx = do
  -- inmx == di, filterd = fd
  r <- Data.BlasM.convolve inmx flt (Proxy :: Proxy fd) (Proxy :: Proxy sx, Proxy :: Proxy sy) (Proxy :: Proxy pxl, Proxy :: Proxy pxr, Proxy :: Proxy pyt, Proxy :: Proxy pyb) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy 0, Proxy :: Proxy 0) False
  -- r' <- add r (cFilterBiases con) -- each filter layer has 1 bias so just need to add it to every output at each depth
  -- need to add each bias in each layer to each layer of output...
  -- r' <- addToAllWithDepth r (cFilterBiases con)
  pure (r, inmx)

-- TODO filter should multiply all weight depth * input depth as well and be run for each filter

conBackward ::
  forall m mx wi hi di fw fh sx sy pxl pxr pyt pyb wo ho fd.
  ( BlasM m mx,
    KnownNat wi,
    KnownNat hi,
    KnownNat di,
    KnownNat fw,
    KnownNat (fw - 1 - pxl),
    KnownNat (fw - 1 - pxr),
    KnownNat fh,
    KnownNat (fh - 1 - pyt),
    KnownNat (fh - 1 - pyb),
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
    KnownNat fd,
    KnownNat (di GHC.TypeLits.* fd),
    Mod (di GHC.TypeLits.* fd) fd ~ 0,
    Div (di GHC.TypeLits.* fd) fd ~ di,
    ((hi + pyt + pyb - fh + sy) ~ (sy GHC.TypeLits.* ho)),
    ((wi + pxl + pxr - fw + sx) ~ (sx GHC.TypeLits.* wo)),
    Mod (wi + pxl + pxr - fw) sx ~ 0,
    Mod (hi + pyt + pyb - fh) sy ~ 0,
    -- verified true for convolve dy w
    (((((ho + ((ho - 1) GHC.TypeLits.* (sy - 1))) + (fh - 1 - pyt)) + (fh - 1 - pyb)) - fh) + 1) ~ hi,
    (((((wo + ((wo - 1) GHC.TypeLits.* (sx - 1))) + (fw - 1 - pxl)) + (fw - 1 - pxr)) - fw) + 1) ~ wi,
    -- verified true for convolve x dy
    ((((hi + pyt) + pyb) - (ho + ((ho - 1) GHC.TypeLits.* (sy - 1)))) + 1) ~ fh,
    ((((wi + pxl) + pxr) - (wo + ((wo - 1) GHC.TypeLits.* (sx - 1)))) + 1) ~ fw
  ) =>
  Proxy '(sx, sy) ->
  Proxy '(pxl, pxr, pyt, pyb) ->
  ConvolveSt mx fw fh fd di ->
  ConvolveIn mx wi hi di ->
  Matrix mx wo ho fd ->
  m (Matrix mx wi hi di, ConvolveG mx fw fh fd di)
conBackward ps pp wgt oldIn dy = do
  -- dy is fd deep
  -- w is fd * id deep
  -- dx is id deep
  ndy <- Data.BlasM.convolveLayersDy dy wgt (Proxy :: Proxy 1, Proxy :: Proxy 1) (Proxy :: Proxy (fw - 1 - pxl), Proxy :: Proxy (fw - 1 - pxr), Proxy :: Proxy (fh - 1 - pyt), Proxy :: Proxy (fh - 1 - pyb)) (Proxy :: Proxy (sx - 1), Proxy :: Proxy (sy - 1)) (Proxy :: Proxy 0, Proxy :: Proxy 0) True
  -- dy has same # of layers as x but we need to output x * fd layers
  dLdW <- Data.BlasM.convolveLayers oldIn dy (Proxy :: Proxy 1, Proxy :: Proxy 1) (Proxy :: Proxy pxl, Proxy :: Proxy pxr, Proxy :: Proxy pyt, Proxy :: Proxy pyb) (Proxy :: Proxy 0, Proxy :: Proxy 0) (Proxy :: Proxy (sx - 1), Proxy :: Proxy (sy - 1)) False

  pure (ndy, dLdW)

conAvg :: (BlasM m mx, KnownNat fw, KnownNat fh, KnownNat fd, KnownNat id, KnownNat (id GHC.TypeLits.* fd)) => Proxy '(fw, fh, fd, id) -> [ConvolveG mx fw fh fd id] -> m (ConvolveG mx fw fh fd id)
conAvg _ gds = avgMxs gds

conUpd :: forall m mx fw fh fd id mod igm. (BlasM m mx, KnownNat fw, KnownNat fh, KnownNat fd, KnownNat id, KnownNat (id GHC.TypeLits.* fd), GradientMod m igm mod (ConvolveG mx fw fh fd id)) => Proxy '(fw, fh, fd, id) -> ConvolveSt mx fw fh fd id -> ConvolveG mx fw fh fd id -> igm -> Maybe mod -> m (ConvolveSt mx fw fh fd id, mod)
conUpd _ wgt dw igm mod = do
  (dw', mod') <- modGradient igm mod dw
  dwOut <- subtractM wgt dw'
  pure (dwOut, mod')

conInit ::
  forall g m mx dpo fw fh di.
  (RandomGen g, BlasM m mx, KnownNat dpo, KnownNat fw, KnownNat fh, KnownNat di, KnownNat (di GHC.TypeLits.* dpo)) =>
  Proxy '(fw, fh, dpo, di) ->
  (g -> (Double, g)) ->
  g ->
  m (ConvolveSt mx fw fh dpo di, g)
conInit _ rf gen =
  let (filt, gen1) = netRandoms rf gen (fromIntegral $ natVal (Proxy :: Proxy fw) * natVal (Proxy :: Proxy fh) * natVal (Proxy :: Proxy (di GHC.TypeLits.* dpo)))
   in do
        fmx <- mxFromList filt (Proxy :: Proxy fw) (Proxy :: Proxy fh) (Proxy :: Proxy (di GHC.TypeLits.* dpo))
        pure $ (fmx, gen1)

convolve ::
  forall m mx wi hi di wo ho g fw fh fd sx sy pxl pxr pyt pyb amod bmod igm.
  ( BlasM m mx,
    KnownNat wi,
    KnownNat hi,
    KnownNat di,
    KnownNat wo,
    KnownNat ho,
    RandomGen g,
    KnownNat fw,
    KnownNat fh,
    KnownNat fd,
    KnownNat sx,
    KnownNat sy,
    KnownNat pxl,
    KnownNat pxr,
    KnownNat pyt,
    KnownNat pyb,
    KnownNat (fw - 1 - pxl),
    KnownNat (fw - 1 - pxr),
    KnownNat (fh - 1 - pyt),
    KnownNat (fh - 1 - pyb),
    KnownNat (sy - 1),
    KnownNat (sx - 1),
    KnownNat (ho - 1),
    KnownNat (wo - 1),
    KnownNat (hi - 1),
    KnownNat (wi - 1),
    KnownNat (di GHC.TypeLits.* fd),
    Mod (di GHC.TypeLits.* fd) fd ~ 0,
    Div (di GHC.TypeLits.* fd) fd ~ di,
    -- no dialation standard convolve size with pad and stride
    ((hi + pyt + pyb - fh + sy) ~ (sy GHC.TypeLits.* ho)),
    ((wi + pxl + pxr - fw + sx) ~ (sx GHC.TypeLits.* wo)),
    -- ensure this is evenly divisible
    Mod (wi + pxl + pxr - fw) sx ~ 0,
    Mod (hi + pyt + pyb - fh) sy ~ 0,
    -- verified true for convolve dy w
    (((((ho + ((ho - 1) GHC.TypeLits.* (sy - 1))) + (fh - 1 - pyt)) + (fh - 1 - pyb)) - fh) + 1) ~ hi,
    (((((wo + ((wo - 1) GHC.TypeLits.* (sx - 1))) + (fw - 1 - pxl)) + (fw - 1 - pxr)) - fw) + 1) ~ wi,
    -- verified true for convolve x dy
    ((((hi + pyt) + pyb) - (ho + ((ho - 1) GHC.TypeLits.* (sy - 1)))) + 1) ~ fh,
    ((((wi + pxl) + pxr) - (wo + ((wo - 1) GHC.TypeLits.* (sx - 1)))) + 1) ~ fw,
    GradientMod m igm amod (ConvolveG mx fw fh fd di),
    GradientMod m igm bmod (Matrix mx 1 1 fd)
  ) =>
  -- | filter size, the filter depth specified matches the # of output layers
  Proxy '(fw, fh, fd) ->
  -- | Stride size
  Proxy '(sx, sy) ->
  -- | Padding size
  Proxy '(pxl, pxr, pyt, pyb) ->
  -- | filter depth is dpo / di
  Layer m mx (ConvolveSt mx fw fh fd di, Matrix mx 1 1 fd) (ConvolveIn mx wi hi di, ()) (ConvolveG mx fw fh fd di, Matrix mx 1 1 fd) wi hi di wo ho fd igm (amod, bmod) g
convolve pxf pxs pxp =
  let px = Proxy :: Proxy '(fw, fh, fd, di)
   in (Layer (conForward pxs pxp) (conBackward pxs pxp) (conAvg px) (conUpd px) (conInit px)) `connect` layerBias (Proxy :: Proxy '(wo, ho, fd))
