{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module ML.NNet.Norm where

import Data.BlasM
import Data.Proxy
import Debug.Trace
import GHC.TypeLits
import ML.NNet
import System.Random

-- beta is just a layer bias
--TODO keep track of gamma
data NormSt mx d = NormSt {normGamma :: (Matrix mx 1 1 d), normMean :: Matrix mx 1 1 d, normVar :: Matrix mx 1 1 d, normInvSDev :: Matrix mx 1 1 d}

type NormIn mx w h d =
  NormIn
    { normXHat :: Matrix mx w h d,
      normIn :: Matrix mx w h d,
      normInMean :: Matrix mx 1 1 d,
      normInVar :: Matrix mx 1 1 d
    }

type NormG mx d =
  NormG
    { normGradGamma :: Vector mx 1 1 d
    , normGradInMean :: Matrix mx 1 1 d
    , normGradInVar :: Matrix mx 1 1 d
    }


normF :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => NormSt mx d -> Matrix mx w h d -> m (Matrix mx w h d, NormIn mx w h d)
normF st inp = do
  negmean <- scale (normMean st) (-1)
  subinp <- addToAllWithDepth inp negmean
  xhat <- mult subinp (normInvSDev)
  y <- multToAllWithDepth xhat (normGamma st)
  inpsum <- sumLayers inp
  inmn <- inpsum `applyFunction` Div Value (Const (fromIntegral $ mxWidth inp * mxHeight inp)) -- average each layer
  invinmn <- scale inmn (-1)
  xmu <- addToAllWithDepth inp invinmn
  xmus <- xmu `applyFunction` Exp Value (Const 2.0)
  xmussum <- sumLayers xmus
  invar <- xmussum `applyFunction` Div Value (Const (fromIntegral $ mxWidth inp * mxHeight inp))
  pure (y, NormIn xhat inp inmn invar)

normB :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => NormSt -> NormIn mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, NormG)
normB st nin dy = do
  gy <- multToAllWithDepth dy (normGamma st)
  gg <- (sumLayers =<< mult (normXHat nin) dy)
  pure (gy, NormG gg (normInMean nin) (normInVar nin))

{-
Initialize M1 = x1 and S1 = 0.

For subsequent x‘s, use the recurrence formulas

Mk = Mk-1 + (xk – Mk-1)/k
Sk = Sk-1 + (xk – Mk-1)*(xk – Mk).
-}

normAvg :: Monad m => [NormG] -> m NormG
normAvg xs =
  a <- cellAvgMxs (map normGradGamma xs)
  -- mu = for each sample (1*mn) / length xs
  -- var = for each sample ()
  mn <- cellAvgMxs (map normGradInMean xs)
  var <- 
  pure a

normUpd :: (Monad m, GradientMod m igm mod NormG) => NormSt -> NormG -> igm -> Maybe mod -> m (NormSt, mod)
normUpd st (NormG ggd) igm mod =
  (dg, mod') <- modGradient igm mod ggd
  


  -- calculate new lmean and ldev and ttl?

  pure (NormSt, mod')

norm :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g, GradientMod m igm mod NormG) => Layer m mx NormSt (NormIn mx w h d) NormG w h d w h d igm mod g
norm =
  Layer
