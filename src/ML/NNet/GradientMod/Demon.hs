{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}

module ML.NNet.GradientMod.Demon where

import Data.BlasM
import GHC.TypeLits
import ML.NNet
import Data.Serialize
import Data.Proxy

-- Number of total iterations and underlying algorithm
data Demon a = Demon Int Double (Double -> a)

data DemonSt b = DemonSt Int b

instance (GradientDescentMethod m mx a b w h d) => GradientDescentMethod m mx (Demon a) (DemonSt b) w h d where
  updateWeights = dmod
  serializeMod = serialize

        
demonCalc :: Double -> Int -> Int -> Double
demonCalc minit iter tIter =
  let rs = ((1.0 - (fromIntegral iter / fromIntegral tIter)) * minit) / (1.0 - minit)
   in rs / (1.0 + rs)

serialize :: forall m mx w h d cnf st. (BlasM m mx, KnownNat w, KnownNat h, KnownNat d, GradientDescentMethod m mx cnf st w h d) =>
  Proxy '(Demon cnf, w, h, d) -> (Get (m (Maybe (DemonSt st))), Maybe (DemonSt st) -> m Put)
serialize _ =
    (do
        i <- getMaybeOf getInt32be
        case i of
          Nothing -> pure $ pure Nothing
          Just iter -> do
            getb <- fst (ML.NNet.serializeMod (Proxy :: Proxy '(cnf,w,h,d)))
            pure $ getb >>= \b ->
              case b of
                Nothing -> pure Nothing
                Just b' -> pure (Just (DemonSt (fromIntegral iter) b')) 
    ,\mst -> do
        case mst of
          Nothing -> pure ((putMaybeOf putInt32be) Nothing)
          Just (DemonSt iter b) -> do
            putb <- (snd (ML.NNet.serializeMod (Proxy :: Proxy '(cnf,w,h,d)))) (Just b)
            pure (putMaybeOf putInt32be (Just (fromIntegral iter)) >> putb)
    )  

--   modGradient :: i -> Maybe a -> b -> m (b, a)
dmod :: (BlasM m mx, KnownNat w, KnownNat h, KnownNat d, GradientDescentMethod m mx a b w h d) => Demon a -> Maybe (DemonSt b) -> Matrix mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, DemonSt b)
dmod (Demon titer minit fa) Nothing wgt grad = do
  let nm = demonCalc minit 1 titer
  (wgt', gst) <- ML.NNet.updateWeights (fa nm) Nothing wgt grad
  pure (wgt', DemonSt 1 gst)
dmod (Demon titer minit fa) (Just (DemonSt citer gst)) wgt grad = do
  let nm = demonCalc minit citer titer
  (wgt', gst') <- ML.NNet.updateWeights (fa nm) (Just gst) wgt grad
  pure (wgt', DemonSt (citer + 1) gst')
