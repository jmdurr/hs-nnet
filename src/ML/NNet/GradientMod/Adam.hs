{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
module ML.NNet.GradientMod.Adam where

import Data.BlasM
import GHC.TypeLits
import ML.NNet
import Data.Serialize
import Data.Proxy

data Adam = Adam Double Double Double

data AdamSt mx w h d = AdamSt Int (Matrix mx w h d) (Matrix mx w h d)

-- iter should start at 0

instance (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => GradientDescentMethod m mx Adam (AdamSt mx w h d) w h d where
  updateWeights (Adam rate b1 b2) st wgt grad = do
    adamD rate b1 b2 st wgt grad
  serializeMod _ =
    (do r <- getMaybeOf getInt32be
        case r of
          Nothing -> pure $ pure Nothing
          Just r' -> do
            getmx1 <- deserializeMx (Proxy :: Proxy '(w,h,d))
            getmx2 <- deserializeMx (Proxy :: Proxy '(w,h,d))
            pure $ do
              mx1 <- getmx1
              mx2 <- getmx2
              pure (Just (AdamSt (fromIntegral r') mx1 mx2))
    ,\mst ->
       case mst of
         Nothing -> pure (putMaybeOf putInt32be Nothing)
         Just (AdamSt iter mx1 mx2) -> do
           pmx1 <- serializeMx mx1
           pmx2 <- serializeMx mx2
           pure $ do
             putMaybeOf putInt32be (Just (fromIntegral iter))
             pmx1
             pmx2
    )
        
-- (Maybe state -> Matrix mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, state)) ->
-- Ot - (Nebula*mhat) / (vhat + epsilon)
{- https://mlfromscratch.com/optimizers-explained/#/
   theta - (rate / (sqrt (eps + squared deltas over time))) * current delta
-}
adamD :: (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => Double -> Double -> Double -> Maybe (AdamSt mx w h d) -> Matrix mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, AdamSt mx w h d)
adamD rate b1 b2 Nothing wgt grad = do
  mt <- konst 0.0 -- momentum
  vt <- konst 0.0 -- velocity
  adamD rate b1 b2 (Just (AdamSt 0 mt vt)) wgt grad
adamD rate b1 b2 (Just (AdamSt iter mt vt)) wgt grad = do
  nz <- nearZero

  -- calc momentum
  mt0 <- scale mt b1
  mt2 <- scale grad (1 - b1)
  mt' <- add mt0 mt2

  -- calc velocity
  vt0 <- scale vt b2
  vt1 <- applyFunction grad (Mul (Mul Value Value) (Const (1 - b2)))
  vt' <- add vt0 vt1

  -- mt hat and vt hat
  mthat <- applyFunction mt' (Div Value (Const (1 - (b1 ** fromIntegral (iter + 1)))))
  vthat <- applyFunction vt' (Div Value (Const (1 - (b2 ** fromIntegral (iter + 1)))))
  -- grad - rate * mthat / sqrt (vthat) + e
  numer <- scale mthat (negate rate)
  denom <- applyFunction vthat (Div (Const 1.0) (Add (Sqrt Value) (Const nz)))
  dived <- mult numer denom
  wgt' <- add wgt dived

  pure (wgt', AdamSt (iter + 1) mt' vt')
