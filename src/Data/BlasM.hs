{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Data.BlasM where

import Control.Monad.IO.Class (MonadIO)
import Data.Hashable
import Data.Proxy
import Data.Word (Word64)
import Foreign.Storable
import GHC.Generics (Generic)
import GHC.TypeLits

data Matrix mx w h d where
  Matrix :: (KnownNat w, KnownNat h, KnownNat d) => mx -> Matrix mx w h d

instance (KnownNat w, KnownNat h, KnownNat d, Show mx) => Show (Matrix mx w h d) where
  show (Matrix mx) = "Matrix " <> show (natVal (undefined :: Proxy w), natVal (undefined :: Proxy h), natVal (undefined :: Proxy d)) <> " [" <> show mx <> "]"

type Vector mx n = Matrix mx 1 n 1

data IfExp
  = IfGt MxFunction MxFunction
  | IfLt MxFunction MxFunction
  | IfEq MxFunction MxFunction
  | IfNe MxFunction MxFunction
  deriving (Show, Eq, Generic)

data MxFunction
  = Value
  | Exp MxFunction MxFunction
  | Log MxFunction
  | Ln MxFunction
  | Div MxFunction MxFunction
  | Mul MxFunction MxFunction
  | Add MxFunction MxFunction
  | Sub MxFunction MxFunction
  | Const Double
  | Random Double Double
  | Min MxFunction MxFunction
  | Max MxFunction MxFunction
  | Sinh MxFunction
  | Cosh MxFunction
  | Sqrt MxFunction
  | Tanh MxFunction
  | If IfExp MxFunction MxFunction
  deriving (Show, Eq, Generic)

instance Hashable MxFunction

instance Hashable IfExp

-- | A class for something that provides a blas like interface
class (MonadIO m, MonadFail m, Show mx) => BlasM m mx | m -> mx where
  dot :: (KnownNat h) => Vector mx h -> Vector mx h -> m Double

  -- (h,w) x (w,w1) = (h,w1)
  dense :: (KnownNat w, KnownNat h, KnownNat d, KnownNat w1) => Matrix mx w h d -> Matrix mx w1 w d -> m (Matrix mx w1 h d)

  -- (h1,w)T x (h1,h2) -> (w,h1) x (h1,h2) -> (w,h2)
  denseT1 :: (KnownNat w, KnownNat d, KnownNat h1, KnownNat h2) => Matrix mx w h1 d -> Matrix mx h2 h1 d -> m (Matrix mx h2 w d)

  -- (h1,w) x (h2,w)T = (h1,w) x (w,h2) -> (h1,h2) -- need to add restrictions?
  denseT2 :: (KnownNat w, KnownNat d, KnownNat h1, KnownNat h2) => Matrix mx w h1 d -> Matrix mx w h2 d -> m (Matrix mx h2 h1 d)

  --denseT2 :: (KnownNat w, KnownNat h, KnownNat d, KnownNat h1) => Matrix mx w h d -> Matrix mx h h1 d -> m (Matrix mx h h1 d)

  -- T(o,i) * (1,o) = (i,o)  w = c, h = r       i  o      o 1    (i,o)
  outer :: (KnownNat l, KnownNat r) => Vector mx l -> Vector mx r -> m (Matrix mx r l 1) -- [l,r]
  scale :: (KnownNat w, KnownNat h, KnownNat d) => Matrix mx w h d -> Double -> m (Matrix mx w h d)
  mult :: (KnownNat w, KnownNat h, KnownNat d) => Matrix mx w h d -> Matrix mx w h d -> m (Matrix mx w h d)
  add ::
    (KnownNat w, KnownNat h, KnownNat d) =>
    Matrix mx w h d ->
    Matrix mx w h d ->
    -- | modifies the second matrix
    m (Matrix mx w h d)
  subtractM ::
    (KnownNat w, KnownNat h, KnownNat d) =>
    Matrix mx w h d ->
    Matrix mx w h d ->
    -- | modifies the second matrix
    m (Matrix mx w h d)
  applyFunction ::
    (KnownNat w, KnownNat h, KnownNat d) =>
    Matrix mx w h d ->
    MxFunction ->
    -- | apply a function to each element of the matrix
    m (Matrix mx w h d)
  konst :: (KnownNat w, KnownNat h, KnownNat d) => Double -> m (Matrix mx w h d)
  reshapeM ::
    forall w h d w2 h2 d2.
    (KnownNat w, KnownNat h, KnownNat d, KnownNat w2, KnownNat h2, KnownNat d2, KnownNat (w GHC.TypeLits.* h), KnownNat (w2 GHC.TypeLits.* h2), (w GHC.TypeLits.* h GHC.TypeLits.* d) ~ (w2 GHC.TypeLits.* h2 GHC.TypeLits.* d2)) =>
    Matrix mx w h d ->
    m (Matrix mx w2 h2 d2)
  mxFromList ::
    (KnownNat w, KnownNat h, KnownNat d) =>
    [Double] ->
    Proxy w ->
    Proxy h ->
    Proxy d ->
    m (Matrix mx w h d)
  mxToLists ::
    (KnownNat w, KnownNat h, KnownNat d) =>
    Matrix mx w h d ->
    m [[[Double]]]
  flip :: (KnownNat w, KnownNat h, KnownNat d) => Matrix mx w h d -> FlipAxis -> m (Matrix mx w h d)

  addToAllWithDepth :: (KnownNat w, KnownNat h, KnownNat d) => Matrix mx w h d -> Matrix mx 1 1 d -> m (Matrix mx w h d)

  sumFlatten :: forall w h d nd. (KnownNat w, KnownNat h, KnownNat d, KnownNat nd) => Matrix mx w h d -> Proxy nd -> m (Matrix mx w h nd)

  sumLayers :: forall w h d. (KnownNat w, KnownNat h, KnownNat d) => Matrix mx w h d -> m (Matrix mx 1 1 d)

  {-
  A filter must always have the same # of layers as input channels
  Therefore multiple filters would have f * l layers and each filter is applied
  to all channels

  This results in a single feature map per filter or 1 output layer per filter

  -}
  convolve ::
    ( KnownNat w,
      KnownNat h,
      KnownNat d,
      KnownNat fw,
      KnownNat fh,
      KnownNat fd,
      KnownNat ow,
      KnownNat oh,
      KnownNat sx,
      KnownNat sy,
      KnownNat pxl,
      KnownNat pxr,
      KnownNat pyt,
      KnownNat pyb,
      KnownNat dx,
      KnownNat dy,
      KnownNat dfx,
      KnownNat dfy,
      KnownNat (h - 1),
      KnownNat (w - 1),
      KnownNat (d GHC.TypeLits.* fd),
      (((h + ((h - 1) GHC.TypeLits.* dy)) + pyt + pyb - (fh + ((fh - 1) GHC.TypeLits.* dfy)) + sy) ~ (sy GHC.TypeLits.* oh)),
      (((w + ((w - 1) GHC.TypeLits.* dx)) + pxl + pxr - (fw + ((fw - 1) GHC.TypeLits.* dfx)) + sx) ~ (sx GHC.TypeLits.* ow)),
      Mod ((w + ((w - 1) GHC.TypeLits.* dx)) + pxl + pxr - (fw + ((fw - 1) GHC.TypeLits.* dfx))) sx ~ 0,
      Mod ((h + ((h - 1) GHC.TypeLits.* dy)) + pyt + pyb - (fh + ((fh - 1) GHC.TypeLits.* dfy))) sy ~ 0
    ) =>
    Matrix mx w h d -> -- 6 + 0 + 0 = 5 ~ 2 * 3
    Matrix mx fw fh (d GHC.TypeLits.* fd) ->
    Proxy fd ->
    -- | stride of filter
    (Proxy sx, Proxy sy) ->
    -- | padding of input each side specified separately for type math
    (Proxy pxl, Proxy pxr, Proxy pyt, Proxy pyb) ->
    -- | dialation of input, 0 is a valid known nat...
    (Proxy dx, Proxy dy) ->
    -- | dialation of the filter
    (Proxy dfx, Proxy dfy) ->
    -- | should flip filter on both x and y axis
    Bool ->
    m (Matrix mx ow oh fd)

  {-
    this convolves individual layers such that, f1 * i1 = o1, f1 * i2 = o2, f2 * i1 = o3, etc
    used for backprop of dw
    input is id fd
    output depth is id * fd
  -}
  convolveLayers ::
    ( KnownNat w,
      KnownNat h,
      KnownNat d,
      KnownNat fw,
      KnownNat fh,
      KnownNat fd,
      KnownNat ow,
      KnownNat oh,
      KnownNat sx,
      KnownNat sy,
      KnownNat pxl,
      KnownNat pxr,
      KnownNat pyt,
      KnownNat pyb,
      KnownNat dx,
      KnownNat dy,
      KnownNat dfx,
      KnownNat dfy,
      KnownNat (h - 1),
      KnownNat (w - 1),
      KnownNat (d GHC.TypeLits.* fd),
      (((h + ((h - 1) GHC.TypeLits.* dy)) + pyt + pyb - (fh + ((fh - 1) GHC.TypeLits.* dfy)) + sy) ~ (sy GHC.TypeLits.* oh)),
      (((w + ((w - 1) GHC.TypeLits.* dx)) + pxl + pxr - (fw + ((fw - 1) GHC.TypeLits.* dfx)) + sx) ~ (sx GHC.TypeLits.* ow)),
      Mod ((w + ((w - 1) GHC.TypeLits.* dx)) + pxl + pxr - (fw + ((fw - 1) GHC.TypeLits.* dfx))) sx ~ 0,
      Mod ((h + ((h - 1) GHC.TypeLits.* dy)) + pyt + pyb - (fh + ((fh - 1) GHC.TypeLits.* dfy))) sy ~ 0
    ) =>
    Matrix mx w h d -> -- 6 + 0 + 0 = 5 ~ 2 * 3
    Matrix mx fw fh fd ->
    -- | stride of filter
    (Proxy sx, Proxy sy) ->
    -- | padding of input each side specified separately for type math
    (Proxy pxl, Proxy pxr, Proxy pyt, Proxy pyb) ->
    -- | dialation of input, 0 is a valid known nat...
    (Proxy dx, Proxy dy) ->
    -- | dialation of the filter
    (Proxy dfx, Proxy dfy) ->
    -- | should flip filter on both x and y axis
    Bool ->
    m (Matrix mx ow oh (d GHC.TypeLits.* fd))

  {-
      this convolves individual layers for backward prop of dy
    -}
  convolveLayersDy ::
    ( KnownNat w,
      KnownNat h,
      KnownNat d,
      KnownNat fw,
      KnownNat fh,
      KnownNat fd,
      KnownNat ow,
      KnownNat oh,
      KnownNat sx,
      KnownNat sy,
      KnownNat pxl,
      KnownNat pxr,
      KnownNat pyt,
      KnownNat pyb,
      KnownNat dx,
      KnownNat dy,
      KnownNat dfx,
      KnownNat dfy,
      KnownNat (h - 1),
      KnownNat (w - 1),
      KnownNat (Div fd d),
      Mod fd d ~ 0,
      (((h + ((h - 1) GHC.TypeLits.* dy)) + pyt + pyb - (fh + ((fh - 1) GHC.TypeLits.* dfy)) + sy) ~ (sy GHC.TypeLits.* oh)),
      (((w + ((w - 1) GHC.TypeLits.* dx)) + pxl + pxr - (fw + ((fw - 1) GHC.TypeLits.* dfx)) + sx) ~ (sx GHC.TypeLits.* ow)),
      Mod ((w + ((w - 1) GHC.TypeLits.* dx)) + pxl + pxr - (fw + ((fw - 1) GHC.TypeLits.* dfx))) sx ~ 0,
      Mod ((h + ((h - 1) GHC.TypeLits.* dy)) + pyt + pyb - (fh + ((fh - 1) GHC.TypeLits.* dfy))) sy ~ 0
    ) =>
    Matrix mx w h d -> -- 6 + 0 + 0 = 5 ~ 2 * 3
    Matrix mx fw fh fd ->
    -- | stride of filter
    (Proxy sx, Proxy sy) ->
    -- | padding of input each side specified separately for type math
    (Proxy pxl, Proxy pxr, Proxy pyt, Proxy pyb) ->
    -- | dialation of input, 0 is a valid known nat...
    (Proxy dx, Proxy dy) ->
    -- | dialation of the filter
    (Proxy dfx, Proxy dfy) ->
    -- | should flip filter on both x and y axis
    Bool ->
    m (Matrix mx ow oh (Div fd d))

-- TODO add a requirement that the filter center is always over the input matrix (not over the padding)
data FlipAxis = FlipX | FlipY | FlipBoth deriving (Eq, Show)

avgMxs :: (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => [Matrix mx w h d] -> m (Matrix mx w h d)
avgMxs [] = error "cannot average empty list"
avgMxs (v : vs) = go v vs
  where
    go v' [] = applyFunction v' (Div Value (Const (fromIntegral $ length (v : vs))))
    go v' (vn : vs') = do
      nv <- add v' vn
      go nv vs'
