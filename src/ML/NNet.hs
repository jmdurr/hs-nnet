{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE UndecidableInstances #-}

module ML.NNet where

import Control.Monad.IO.Class (liftIO)
import Data.BlasM
import qualified Data.ByteString.Lazy as BL
import Data.List (intersperse)
import Data.Proxy
import Data.Serialize
import Data.Text
import qualified Data.Vector.Storable as V
import GHC.TypeLits
import ML.NNet.Init.RandomFun
import System.Random
import Text.Printf
import Prelude hiding (init)

class (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => GradientDescentMethod m mx conf state w h d | conf mx w h d -> state where
  updateWeights ::
    (conf -> Maybe state -> Matrix mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, state))
  serializeMod :: Proxy '(conf, w, h, d) -> (Get (m (Maybe state)), Maybe state -> m Put)

-- | Layer has an input and an output
data Layer m mx lst ist gd wi hi di wo ho dpo gconf gmod g where
  Layer ::
    (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo, KnownNat ho, KnownNat dpo, RandomGen g) =>
    -- | forward prop
    (lst -> Matrix mx wi hi di -> m (Matrix mx wo ho dpo, ist)) ->
    -- | backward prop
    (lst -> ist -> Matrix mx wo ho dpo -> m (Matrix mx wi hi di, gd)) ->
    -- | gradient accum and divide
    (gd -> gd -> m gd, gd -> Int -> m gd) ->
    -- | gradient update, uses GradientDescentMethod gmod
    (gconf -> lst -> gd -> m lst) ->
    -- | init function
    (WeightInitializer g -> g -> m (lst, g)) ->
    -- | Serialize and deserialize with optional learning state
    (gconf -> (Get (m lst), lst -> m Put)) ->
    Layer m mx lst ist gd wi hi di wo ho dpo gconf gmod g

connectForward :: Monad m => Layer m mx alst aist gda wi hi di wo ho dpo agconf agmod g -> Layer m mx blst bist gdb wo ho dpo wo2 ho2 dpo2 bgconf bgmod g -> (alst, blst) -> Matrix mx wi hi di -> m (Matrix mx wo2 ho2 dpo2, (aist, bist))
connectForward (Layer l1f _ _ _ _ _) (Layer l2f _ _ _ _ _) (a, b) mx = do
  -- forward a mx
  -- forward b mx
  (mx', a') <- l1f a mx
  --liftIO $ putStrLn "init 2"
  (mxo, b') <- l2f b mx'
  pure (mxo, (a', b'))

connectBackward :: Monad m => Layer m mx alst aist gda wi hi di wo ho dpo agconf agmod g -> Layer m mx blst bist gdb wo ho dpo wo2 ho2 dpo2 bgconf bgmod g -> (alst, blst) -> (aist, bist) -> Matrix mx wo2 ho2 dpo2 -> m (Matrix mx wi hi di, (gda, gdb))
connectBackward (Layer _ l2a _ _ _ _) (Layer _ l2b _ _ _ _) (a, b) (ai, bi) mx = do
  (mx', b') <- l2b b bi mx
  (mxo, a') <- l2a a ai mx'
  pure (mxo, (a', b'))

connectInit :: (RandomGen g, Monad m) => Layer m mx alst aist gda wi hi di wo ho dpo agconf agmod g -> Layer m mx blst bist gdb wo ho dpo wo2 ho2 dpo2 bgconf bgmod g -> (WeightInitializer g -> g -> m ((alst, blst), g))
connectInit (Layer _ _ _ _ l1 _) (Layer _ _ _ _ l2 _) f gen = do
  --liftIO $ putStrLn "init 1"
  (a, g') <- l1 f gen
  --liftIO $ putStrLn "init 2"
  (b, g2) <- l2 f g'
  pure ((a, b), g2)

connectUpdate :: (Monad m) => Layer m mx alst aist gda wi hi di wo ho dpo conf gst g -> Layer m mx blst bist gdb wo ho dpo wo2 ho2 dpo2 conf gbst g -> conf -> (alst, blst) -> (gda, gdb) -> m (alst, blst)
connectUpdate (Layer _ _ _ l1 _ _) (Layer _ _ _ l2 _ _) c (a, b) (ga, gb) =
  do
    bst' <- l2 c b gb
    ast' <- l1 c a ga
    pure (ast', bst')

--(Maybe state -> Matrix mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, state)) ->

connectAverage :: Monad m => Layer m mx alst aist gda wi hi di wo ho dpo agconf agmod g -> Layer m mx blst bist gdb wo ho dpo wo2 ho2 dpo2 bgconf bgmod g -> ((gda, gdb) -> (gda, gdb) -> m (gda, gdb), (gda, gdb) -> Int -> m (gda, gdb))
connectAverage (Layer _ _ (accu1, div1) _ _ _) (Layer _ _ (accu2, div2) _ _ _) =
  ( \(gdal, gdbl) (gdar, gdbr) -> do
      gda <- accu1 gdal gdar
      gdb <- accu2 gdbl gdbr
      pure (gda, gdb),
    \(gdal, gdbl) i -> do
      gda <- div1 gdal i
      gdb <- div2 gdbl i
      pure (gda, gdb)
  )

connectSerialize :: Monad m => Layer m mx alst aist gda wi hi di wo ho dpo conf agmod g -> Layer m mx blst bist gdb wo ho dpo wo2 ho2 dpo2 conf bgmod g -> (conf -> (Get (m (alst, blst)), (alst, blst) -> m Put))
connectSerialize (Layer _ _ _ _ _ sa) (Layer _ _ _ _ _ sb) c =
  ( do
      ma <- fst (sa c)
      mb <- fst (sb c)
      pure $ do
        alst' <- ma
        blst' <- mb
        pure (alst', blst'),
    \(alst, blst) -> do
      p1 <- (snd (sa c)) alst
      p2 <- (snd (sb c)) blst
      pure (p1 >> p2)
  )

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
    (connectSerialize l1 l2)

data Network m mx alst aist gd wi hi di wo2 ho2 dpo2 gconf mod g = Network (Layer m mx alst aist gd wi hi di wo2 ho2 dpo2 gconf mod g) gconf

initNetwork ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo, KnownNat ho, KnownNat dpo, RandomGen g) =>
  g ->
  WeightInitializer g ->
  Layer m mx alst aist gd wi hi di wo ho dpo gconf mod g ->
  gconf ->
  m (Network m mx alst aist gd wi hi di wo ho dpo gconf mod g, alst, g)
initNetwork gen f l@(Layer _ _ _ _ init _) gc = do
  (a, g') <- init f gen
  pure (Network l gc, a, g')

-- runForward gives the outputs
runForward ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo2, KnownNat ho2, KnownNat dpo2, RandomGen g) =>
  Network m mx alst aist gd wi hi di wo2 ho2 dpo2 gconf mod g ->
  alst ->
  Matrix mx wi hi di ->
  m (Matrix mx wo2 ho2 dpo2, aist)
runForward (Network (Layer for _ _ _ _ _) _) a mx = for a mx

runBackward ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo2, KnownNat ho2, KnownNat dpo2, RandomGen g) =>
  Network m mx alst aist gd wi hi di wo2 ho2 dpo2 gconf mod g ->
  alst ->
  aist ->
  Matrix mx wo2 ho2 dpo2 ->
  m (Matrix mx wi hi di, gd)
runBackward (Network (Layer _ bak _ _ _ _) _) a ai dErr = bak a ai dErr

runUpdate ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo2, KnownNat ho2, KnownNat dpo2, RandomGen g) =>
  Network m mx alst aist gd wi hi di wo2 ho2 dpo2 gconf mod g ->
  alst ->
  gd ->
  m alst
runUpdate (Network (Layer _ _ _ upd _ _) gc) a grad = upd gc a grad

accumulateGradients ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo2, KnownNat ho2, KnownNat dpo2, RandomGen g) =>
  Network m mx alst aist gd wi hi di wo2 ho2 dpo2 gconf mod g ->
  gd ->
  gd ->
  m gd
accumulateGradients (Network (Layer _ _ (accu, _) _ _ _) _) = accu

divideGradients ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo2, KnownNat ho2, KnownNat dpo2, RandomGen g) =>
  Network m mx alst aist gd wi hi di wo2 ho2 dpo2 gconf mod g ->
  gd ->
  Int ->
  m gd
divideGradients (Network (Layer _ _ (_, divide) _ _ _) _) = divide

netRandoms :: WeightInitializer g -> g -> Int -> Int -> Int -> ([Double], g)
netRandoms _ gen 0 _ _ = ([], gen)
netRandoms f gen num fanIn fanOut =
  let (d, g') = f fanIn fanOut gen
      (ds, gs) = netRandoms f g' (num - 1) fanIn fanOut
   in (d : ds, gs)

randomMx :: (BlasM m mx, KnownNat w, KnownNat h, KnownNat d, RandomGen g) => WeightInitializer g -> g -> Proxy w -> Proxy h -> Proxy d -> Int -> Int -> m (Matrix mx w h d, g)
randomMx f g pw ph pd fanIn fanOut =
  let w = fromIntegral $ natVal pw
      h = fromIntegral $ natVal ph
      d = fromIntegral $ natVal pd
      (ls, g') = netRandoms f g (w * h * d) fanIn fanOut
   in do
        --liftIO $ putStrLn $ "NN rmx " <> show w <> ":" <> show h <> ":" <> show d <> ":" <> show (length ls)
        mx <- mxFromVec (V.fromList ls) pw ph pd
        pure (mx, g')

mxToCSV :: (BlasM m mx, KnownNat w, KnownNat h, KnownNat d) => Matrix mx w h d -> m Text
mxToCSV mx =
  let rowToCsv row = Data.Text.concat (Data.List.intersperse (pack ",") (Prelude.map (pack . printf "%.3f") row)) <> pack "\n"
      colToCsv col = Data.Text.concat (Prelude.map rowToCsv col) <> pack "\n"
      mxToCsv ls = Data.Text.concat (Prelude.map colToCsv ls)
   in do
        ls <- mxToLists mx -- [[[row],[row],[row]],[[row]...]
        pure (mxToCsv ls)

serializeNet ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo2, KnownNat ho2, KnownNat dpo2, RandomGen g) =>
  Network m mx alst aist gd wi hi di wo2 ho2 dpo2 gconf mod g ->
  alst ->
  m BL.ByteString
serializeNet (Network (Layer _ _ _ _ _ gs) conf) lst = do
  pf <- (snd (gs conf)) lst
  pure $ runPutLazy pf

deserializeNet ::
  (BlasM m mx, KnownNat wi, KnownNat hi, KnownNat di, KnownNat wo2, KnownNat ho2, KnownNat dpo2, RandomGen g) =>
  Network m mx alst aist gd wi hi di wo2 ho2 dpo2 gconf mod g ->
  BL.ByteString ->
  m alst
deserializeNet (Network (Layer _ _ _ _ _ gs) conf) bs = do
  case runGetLazy (fst (gs conf)) bs of
    Left e -> fail e
    Right v -> v
