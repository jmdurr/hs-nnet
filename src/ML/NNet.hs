{-# LANGUAGE DataKinds #-}
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
import Data.Proxy
import GHC.TypeLits
import System.Random

-- | Method to apply various gradient techniques
class (Monad m) => GradientMod m i a b | i b -> a where
  modGradient :: i -> Maybe a -> b -> m (b, a)

instance (Monad m, GradientMod m i a b, GradientMod m i c d) => GradientMod m i (a, c) (b, d) where
  modGradient i Nothing (b, d) = do
    (b', a) <- modGradient i Nothing b
    (d', c) <- modGradient i Nothing d
    pure ((b', d'), (a, c))
  modGradient i (Just (a, c)) (b, d) = do
    (b', a') <- modGradient i (Just a) b
    (d', c') <- modGradient i (Just c) d
    pure ((b', d'), (a', c'))

-- | Layer has an input and an output
data Layer m mx lst ist gd wi hi di wo ho dpo igm gmod g where
  Layer ::
    (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo, KnownNat ho, KnownNat dpo, RandomGen g, GradientMod m igm gmod gd) =>
    -- | forward prop
    (lst -> Matrix mx wi hi di -> m (Matrix mx wo ho dpo, ist)) ->
    -- | backward prop
    (lst -> ist -> Matrix mx wo ho dpo -> m (Matrix mx wi hi di, gd)) ->
    -- | gradient average
    ([gd] -> m gd) ->
    -- | gradient update
    (lst -> gd -> igm -> Maybe gmod -> m (lst, gmod)) ->
    -- | init function
    ((g -> (Double, g)) -> g -> m (lst, g)) ->
    Layer m mx lst ist gd wi hi di wo ho dpo igm gmod g

connectForward :: Monad m => Layer m mx alst aist gda wi hi di wo ho dpo igm agmod g -> Layer m mx blst bist gdb wo ho dpo wo2 ho2 dpo2 igm bgmod g -> (alst, blst) -> Matrix mx wi hi di -> m (Matrix mx wo2 ho2 dpo2, (aist, bist))
connectForward (Layer l1f _ _ _ _) (Layer l2f _ _ _ _) (a, b) mx = do
  -- forward a mx
  -- forward b mx
  (mx', a') <- l1f a mx
  --liftIO $ putStrLn "init 2"
  (mxo, b') <- l2f b mx'
  pure (mxo, (a', b'))

connectBackward :: Monad m => Layer m mx alst aist gda wi hi di wo ho dpo igm agmod g -> Layer m mx blst bist gdb wo ho dpo wo2 ho2 dpo2 igm bgmod g -> (alst, blst) -> (aist, bist) -> Matrix mx wo2 ho2 dpo2 -> m (Matrix mx wi hi di, (gda, gdb))
connectBackward (Layer _ l2a _ _ _) (Layer _ l2b _ _ _) (a, b) (ai, bi) mx = do
  (mx', b') <- l2b b bi mx
  (mxo, a') <- l2a a ai mx'
  pure (mxo, (a', b'))

connectInit :: (RandomGen g, Monad m) => Layer m mx alst aist gda wi hi di wo ho dpo igm agmod g -> Layer m mx blst bist gdb wo ho dpo wo2 ho2 dpo2 igm bgmod g -> ((g -> (Double, g)) -> g -> m ((alst, blst), g))
connectInit (Layer _ _ _ _ l1) (Layer _ _ _ _ l2) f gen = do
  --liftIO $ putStrLn "init 1"
  (a, g') <- l1 f gen
  --liftIO $ putStrLn "init 2"
  (b, g2) <- l2 f g'
  pure ((a, b), g2)

connectUpdate :: (GradientMod m igm amod gda, GradientMod m igm bmod gdb, Monad m) => Layer m mx alst aist gda wi hi di wo ho dpo igm amod g -> Layer m mx blst bist gdb wo ho dpo wo2 ho2 dpo2 igm bmod g -> (alst, blst) -> (gda, gdb) -> igm -> Maybe (amod, bmod) -> m ((alst, blst), (amod, bmod))
connectUpdate (Layer _ _ _ l1 _) (Layer _ _ _ l2 _) (a, b) (ga, gb) igm mods =
  do
    (b', bmod') <- l2 b gb igm (snd <$> mods)
    (a', amod') <- l1 a ga igm (fst <$> mods)
    pure ((a', b'), (amod', bmod'))

connectAverage :: Monad m => Layer m mx alst aist gda wi hi di wo ho dpo igm agmod g -> Layer m mx blst bist gdb wo ho dpo wo2 ho2 dpo2 igm bgmod g -> [(gda, gdb)] -> m (gda, gdb)
connectAverage (Layer _ _ l1 _ _) (Layer _ _ l2 _ _) gs =
  let (gdas, gdbs) = unzip gs
   in do
        ma <- l1 gdas
        mb <- l2 gdbs
        pure (ma, mb)

connect ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo, KnownNat ho, KnownNat dpo, KnownNat wo2, KnownNat ho2, KnownNat dpo2, RandomGen g, GradientMod m igm agmod gda, GradientMod m igm bgmod gdb) =>
  Layer m mx alst aist gda wi hi di wo ho dpo igm agmod g ->
  Layer m mx blst bist gdb wo ho dpo wo2 ho2 dpo2 igm bgmod g ->
  Layer m mx (alst, blst) (aist, bist) (gda, gdb) wi hi di wo2 ho2 dpo2 igm (agmod, bgmod) g
connect l1 l2 =
  Layer
    (connectForward l1 l2)
    (connectBackward l1 l2)
    (connectAverage l1 l2)
    (connectUpdate l1 l2)
    (connectInit l1 l2)

data Network m mx alst aist gd wi hi di wo2 ho2 dpo2 igm mod g = Network (Layer m mx alst aist gd wi hi di wo2 ho2 dpo2 igm mod g)

initNetwork ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo, KnownNat ho, KnownNat dpo, RandomGen g) =>
  g ->
  (g -> (Double, g)) ->
  Layer m mx alst aist gd wi hi di wo ho dpo igm mod g ->
  m (Network m mx alst aist gd wi hi di wo ho dpo igm mod g, alst, g)
initNetwork gen f l@(Layer _ _ _ _ init) = do
  (a, g') <- init f gen
  pure (Network l, a, g')

-- runForward gives the outputs
runForward ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo2, KnownNat ho2, KnownNat dpo2, RandomGen g) =>
  Network m mx alst aist gd wi hi di wo2 ho2 dpo2 igm mod g ->
  alst ->
  Matrix mx wi hi di ->
  m (Matrix mx wo2 ho2 dpo2, aist)
runForward (Network l@(Layer for _ _ _ _)) a mx = for a mx

runBackward ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo2, KnownNat ho2, KnownNat dpo2, RandomGen g) =>
  Network m mx alst aist gd wi hi di wo2 ho2 dpo2 igm mod g ->
  alst ->
  aist ->
  Matrix mx wo2 ho2 dpo2 ->
  m (Matrix mx wi hi di, gd)
runBackward (Network l@(Layer _ bak _ _ _)) a ai dErr = bak a ai dErr

runUpdate ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo2, KnownNat ho2, KnownNat dpo2, RandomGen g, GradientMod m igm mod gd) =>
  Network m mx alst aist gd wi hi di wo2 ho2 dpo2 igm mod g ->
  alst ->
  gd ->
  igm ->
  Maybe mod ->
  m (alst, mod)
runUpdate (Network (Layer _ _ _ upd _)) a grad igm m = upd a grad igm m

avgGradients ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo2, KnownNat ho2, KnownNat dpo2, RandomGen g) =>
  Network m mx alst aist gd wi hi di wo2 ho2 dpo2 igm mod g ->
  [gd] ->
  m gd
avgGradients (Network (Layer _ _ avg _ _)) g = avg g

netRandoms :: (g -> (Double, g)) -> g -> Int -> ([Double], g)
netRandoms _ gen 0 = ([], gen)
netRandoms f gen num =
  let (d, g') = f gen
      (ds, gs) = netRandoms f g' (num - 1)
   in (d : ds, gs)

randomMx :: (BlasM m mx, KnownNat w, KnownNat h, KnownNat d, RandomGen g) => (g -> (Double, g)) -> g -> Proxy w -> Proxy h -> Proxy d -> m (Matrix mx w h d, g)
randomMx f g pw ph pd =
  let w = fromIntegral $ natVal pw
      h = fromIntegral $ natVal ph
      d = fromIntegral $ natVal pd
      (ls, g') = netRandoms f g (w * h * d)
   in do
        --liftIO $ putStrLn $ "NN rmx " <> show w <> ":" <> show h <> ":" <> show d <> ":" <> show (length ls)
        mx <- mxFromList ls pw ph pd
        pure (mx, g')
