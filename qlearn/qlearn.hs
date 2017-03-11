import System.Random
import Data.List

data Cell = Empty | Wall | Can | ERob | CRob deriving (Show, Eq)

getCell :: Int -> Cell
getCell n = case n `mod` 10 of
            0 -> Can
            _ -> Empty

randCell :: IO Cell
randCell = do
            n <- randomRIO(1,10)
            return(getCell n)

horizWall :: Int -> [Cell]
horizWall n = [Wall | i<-[1..n]]

internal :: Int -> [Cell]
internal n = let l = [Empty | i<-[1..n-2]] ++ [Wall] in Wall : l

field :: Int -> [[Cell]]
field d = let l = [internal d | i<-[1..d]] ++ [horizWall d] in horizWall d : l

letterOf :: Cell -> String
letterOf Empty = "[ ]"
letterOf Wall = " = "
letterOf Can = "[x]"
letterOf ERob = "[O]"
letterOf CRob = "[Q]"

listValues :: [[Cell]] -> [[String]]
listValues xs = map (map letterOf) xs
--listValues cs = map letterOf cs

printField :: [[Cell]] -> IO ()
--printField xs = print [a | y<-listValues xs, a<-y]
--printField xs = mapM_ putStrLn [a | y<-listValues xs, a<-y]
printField xs = mapM_ putStrLn [ intercalate "" a | a<-listValues xs]

main :: IO ()

main = do
    --n <- randomRIO(1,3) :: IO Int
    --let c = getCell n in print c
    let f = field 10 in printField f



