{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}

module ML.NNet where

import Control.Monad.IO.Class
import Data.BlasM
import qualified Data.ByteString as BS
import Data.Proxy
import GHC.TypeLits
import ML.NNet.Init.RandomFun
import System.Random

class (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => GradientDescentMethod m mx conf state w h d | conf mx w h d -> state where
  updateWeights ::
    (conf -> Maybe state -> Matrix mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, state))

-- | Layer has an input and an output
data Layer m mx lst ist gd wi hi di wo ho dpo gconf gmod g where
  Layer ::
    (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo, KnownNat ho, KnownNat dpo, RandomGen g) =>
    -- | forward prop
    (lst -> Matrix mx wi hi di -> m (Matrix mx wo ho dpo, ist)) ->
    -- | backward prop
    (lst -> ist -> Matrix mx wo ho dpo -> m (Matrix mx wi hi di, gd)) ->
    -- | gradient average
    ([gd] -> m gd) ->
    -- | gradient update, uses GradientDescentMethod gmod
    (gconf -> lst -> gd -> m lst) ->
    -- | init function
    (WeightInitializer g -> g -> m (lst, g)) ->
    Layer m mx lst ist gd wi hi di wo ho dpo gconf gmod g

connectForward :: Monad m => Layer m mx alst aist gda wi hi di wo ho dpo agconf agmod g -> Layer m mx blst bist gdb wo ho dpo wo2 ho2 dpo2 bgconf bgmod g -> (alst, blst) -> Matrix mx wi hi di -> m (Matrix mx wo2 ho2 dpo2, (aist, bist))
connectForward (Layer l1f _ _ _ _) (Layer l2f _ _ _ _) (a, b) mx = do
  -- forward a mx
  -- forward b mx
  (mx', a') <- l1f a mx
  --liftIO $ putStrLn "init 2"
  (mxo, b') <- l2f b mx'
  pure (mxo, (a', b'))

connectBackward :: Monad m => Layer m mx alst aist gda wi hi di wo ho dpo agconf agmod g -> Layer m mx blst bist gdb wo ho dpo wo2 ho2 dpo2 bgconf bgmod g -> (alst, blst) -> (aist, bist) -> Matrix mx wo2 ho2 dpo2 -> m (Matrix mx wi hi di, (gda, gdb))
connectBackward (Layer _ l2a _ _ _) (Layer _ l2b _ _ _) (a, b) (ai, bi) mx = do
  (mx', b') <- l2b b bi mx
  (mxo, a') <- l2a a ai mx'
  pure (mxo, (a', b'))

connectInit :: (RandomGen g, Monad m) => Layer m mx alst aist gda wi hi di wo ho dpo agconf agmod g -> Layer m mx blst bist gdb wo ho dpo wo2 ho2 dpo2 bgconf bgmod g -> (WeightInitializer g -> g -> m ((alst, blst), g))
connectInit (Layer _ _ _ _ l1) (Layer _ _ _ _ l2) f gen = do
  --liftIO $ putStrLn "init 1"
  (a, g') <- l1 f gen
  --liftIO $ putStrLn "init 2"
  (b, g2) <- l2 f g'
  pure ((a, b), g2)

connectUpdate :: (Monad m) => Layer m mx alst aist gda wi hi di wo ho dpo conf gst g -> Layer m mx blst bist gdb wo ho dpo wo2 ho2 dpo2 conf gbst g -> conf -> (alst, blst) -> (gda, gdb) -> m (alst, blst)
connectUpdate (Layer _ _ _ l1 _) (Layer _ _ _ l2 _) c (a, b) (ga, gb) =
  do
    bst' <- l2 c b gb
    ast' <- l1 c a ga
    pure (ast', bst')

--(Maybe state -> Matrix mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, state)) ->

connectAverage :: Monad m => Layer m mx alst aist gda wi hi di wo ho dpo agconf agmod g -> Layer m mx blst bist gdb wo ho dpo wo2 ho2 dpo2 bgconf bgmod g -> [(gda, gdb)] -> m (gda, gdb)
connectAverage (Layer _ _ l1 _ _) (Layer _ _ l2 _ _) gs =
  let (gdas, gdbs) = unzip gs
   in do
        ma <- l1 gdas
        mb <- l2 gdbs
        pure (ma, mb)

connect ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo, KnownNat ho, KnownNat dpo, KnownNat wo2, KnownNat ho2, KnownNat dpo2, RandomGen g) =>
  Layer m mx alst aist gda wi hi di wo ho dpo conf gst g ->
  Layer m mx blst bist gdb wo ho dpo wo2 ho2 dpo2 conf gbst g ->
  Layer m mx (alst, blst) (aist, bist) (gda, gdb) wi hi di wo2 ho2 dpo2 conf (gst, gbst) g
connect l1 l2 =
  Layer
    (connectForward l1 l2)
    (connectBackward l1 l2)
    (connectAverage l1 l2)
    (connectUpdate l1 l2)
    (connectInit l1 l2)

data Network m mx alst aist gd wi hi di wo2 ho2 dpo2 gconf mod g = Network (Layer m mx alst aist gd wi hi di wo2 ho2 dpo2 gconf mod g) gconf

initNetwork ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo, KnownNat ho, KnownNat dpo, RandomGen g) =>
  g ->
  WeightInitializer g ->
  Layer m mx alst aist gd wi hi di wo ho dpo gconf mod g ->
  gconf ->
  m (Network m mx alst aist gd wi hi di wo ho dpo gconf mod g, alst, g)
initNetwork gen f l@(Layer _ _ _ _ init) gc = do
  (a, g') <- init f gen
  pure (Network l gc, a, g')

-- runForward gives the outputs
runForward ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo2, KnownNat ho2, KnownNat dpo2, RandomGen g) =>
  Network m mx alst aist gd wi hi di wo2 ho2 dpo2 gconf mod g ->
  alst ->
  Matrix mx wi hi di ->
  m (Matrix mx wo2 ho2 dpo2, aist)
runForward (Network l@(Layer for _ _ _ _) _) a mx = for a mx

runBackward ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo2, KnownNat ho2, KnownNat dpo2, RandomGen g) =>
  Network m mx alst aist gd wi hi di wo2 ho2 dpo2 gconf mod g ->
  alst ->
  aist ->
  Matrix mx wo2 ho2 dpo2 ->
  m (Matrix mx wi hi di, gd)
runBackward (Network l@(Layer _ bak _ _ _) _) a ai dErr = bak a ai dErr

runUpdate ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo2, KnownNat ho2, KnownNat dpo2, RandomGen g) =>
  Network m mx alst aist gd wi hi di wo2 ho2 dpo2 gconf mod g ->
  alst ->
  gd ->
  m alst
runUpdate (Network (Layer _ _ _ upd _) gc) a grad = upd gc a grad

avgGradients ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo2, KnownNat ho2, KnownNat dpo2, RandomGen g) =>
  Network m mx alst aist gd wi hi di wo2 ho2 dpo2 gconf mod g ->
  [gd] ->
  m gd
avgGradients (Network (Layer _ _ avg _ _) _) g = avg g

netRandoms :: WeightInitializer g -> g -> Int -> Int -> Int -> ([Double], g)
netRandoms _ gen 0 _ _ = ([], gen)
netRandoms f gen num fanIn fanOut =
  let (d, g') = f fanIn fanOut gen
      (ds, gs) = netRandoms f g' (num - 1) fanIn fanOut
   in (d : ds, gs)

randomMx :: (BlasM m mx, KnownNat w, KnownNat h, KnownNat d, RandomGen g) => WeightInitializer g -> g -> Proxy w -> Proxy h -> Proxy d -> m (Matrix mx w h d, g)
randomMx f g pw ph pd =
  let w = fromIntegral $ natVal pw
      h = fromIntegral $ natVal ph
      d = fromIntegral $ natVal pd
      (ls, g') = netRandoms f g (w * h * d) 0 0
   in do
        --liftIO $ putStrLn $ "NN rmx " <> show w <> ":" <> show h <> ":" <> show d <> ":" <> show (length ls)
        mx <- mxFromList ls pw ph pd
        pure (mx, g')
