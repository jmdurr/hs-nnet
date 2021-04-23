{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module ML.NNet.LayerBias where

-- one bias per layer as used by convolve and deconvolve

import Control.Monad.IO.Class (liftIO)
import Data.BlasM
import Data.Proxy
import Data.Serialize
import GHC.TypeLits
import ML.NNet
import ML.NNet.Init.RandomFun
import System.Random

data LayerBiasSt mx d mod = LayerBiasSt (Matrix mx 1 1 d) (Maybe mod)

layerBiasSerialize :: forall m mx d conf mod. (KnownNat d, BlasM m mx, GradientDescentMethod m mx conf mod 1 1 d) => conf -> (Get (m (LayerBiasSt mx d mod)), LayerBiasSt mx d mod -> m Put)
layerBiasSerialize _ =
  ( do
      mx <- deserializeMx (Proxy :: Proxy '(1, 1, d))
      md <- fst $ serializeMod (Proxy :: Proxy '(conf, 1, 1, d))
      pure $ do
        mxv <- mx
        mdv <- md
        pure (LayerBiasSt mxv mdv),
    \(LayerBiasSt mx md) -> do
      p1 <- serializeMx mx
      p2 <- (snd (serializeMod (Proxy :: Proxy '(conf, 1, 1, d)))) md
      pure (p1 >> p2)
  )

layerBiasForward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => LayerBiasSt mx d mod -> Matrix mx w h d -> m (Matrix mx w h d, ())
layerBiasForward (LayerBiasSt b _) mx = do
  -- add Vector[l] to values in each layer l
  mx' <- addToAllWithDepth mx b
  pure (mx', ())

layerBiasBackward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => LayerBiasSt mx d mod -> () -> Matrix mx w h d -> m (Matrix mx w h d, Matrix mx 1 1 d)
layerBiasBackward _ _ dz = do
  b' <- sumLayers dz
  pure (dz, b')

layerBiasAverageGrad :: (BlasM m mx, KnownNat d, Monad m) => (Matrix mx 1 1 d -> Matrix mx 1 1 d -> m (Matrix mx 1 1 d), Matrix mx 1 1 d -> Int -> m (Matrix mx 1 1 d))
layerBiasAverageGrad = (add, \m i -> scale m (1.0 / fromIntegral i))

layerBiasUpdate :: (Monad m, BlasM m mx, KnownNat d, GradientDescentMethod m mx conf mod 1 1 d) => conf -> LayerBiasSt mx d mod -> Matrix mx 1 1 d -> m (LayerBiasSt mx d mod)
layerBiasUpdate conf (LayerBiasSt bias modst) dbias = do
  (wgt', modst') <- updateWeights conf modst bias dbias

  --  liftIO $ putStrLn $ "update layerbias weights from " <> show dwgt <> " to " <> show dwgt'
  pure (LayerBiasSt wgt' (Just modst'))

--Layer m mx (ConvolveSt mx fw fh fd di) (ConvolveIn mx wi hi di) (ConvolveG mx fw fh fd di) wi hi di wo ho fd igm mod g
-- (FFN a i o) (FFNIn a i) (FFNGrad a i o)
layerBias :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g, GradientDescentMethod m mx conf igm 1 1 d) => Proxy '(w, h, d) -> Layer m mx (LayerBiasSt mx d igm) () (Matrix mx 1 1 d) w h d w h d conf igm g
layerBias _ = Layer layerBiasForward layerBiasBackward layerBiasAverageGrad layerBiasUpdate layerBiasInit layerBiasSerialize

layerBiasInit :: forall m mx g d mod. (Monad m, RandomGen g, BlasM m mx, KnownNat d) => WeightInitializer g -> g -> m (LayerBiasSt mx d mod, g)
layerBiasInit _ g = konst 0.0 >>= \m -> pure (LayerBiasSt m Nothing, g)

{-
let (b', gen1) = netRandoms rf g (fromIntegral $ natVal (Proxy :: Proxy d))
  in do
       b <- mxFromList b' (Proxy :: Proxy 1) (Proxy :: Proxy 1) (Proxy :: Proxy d)
       pure (LayerBiasSt b Nothing, gen1)
 -}
