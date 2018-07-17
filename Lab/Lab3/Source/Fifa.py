import os
os.environ["SPARK_HOME"] = "C:\spark"
os.environ["HADOOP_HOME"] = "C:\winutils"

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("Pyspark SQL Lab3") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

from operator import add

# 1 - Create Spark DataFrames
resultsDf = spark.read.load("C:\\Users\\amehta\\PycharmProjects\\Spark\\WorldCups.csv",
                     format="csv", sep=",", inferSchema="true", header="true")
matchesDf = spark.read.load("C:\\Users\\amehta\\PycharmProjects\\Spark\\WorldCupMatches.csv",
                     format="csv", sep=",", inferSchema="true", header="true")

resultsDf.createOrReplaceTempView("results")
matchesDf.createOrReplaceTempView("matches")

print("Task 1 - Created dataframes\n")

# 2 - Perform 10 intuitive queries (Using Dataframes and SQL)

print("Task 2 - 10 Intuitive Queries")

# a - Total football world cups
total = resultsDf.count()
print("\nNumber of football world cups = " + str(total))

# b - Countries winning world cups in desc order
print("\nNumber of world champions per country")
resultsDf.groupBy("Winner").count().orderBy('count', ascending=False).show()

print("\nNumber of times runners-up per country")
resultsDf.groupBy("Runners-Up").count().orderBy('count', ascending=False).show()

# c - Year when hosting country won the world cup final
print("\nYear when hosting country won the final")
spark.sql("SELECT year, country, winner FROM results where Country = Winner").show()

# d - Average goals scored per match by year
print("\nAverage goal scored per match by year")
spark.sql("SELECT year, CEILING(GoalsScored/MatchesPlayed) avg_goals_per_match FROM results").show()

# e - Match with highest attendance
print("\nMatch with highest attendance")
spark.sql("SELECT Year, Datetime, Stadium, City, `Home Team Name`, `Away Team Name`, Attendance FROM matches "
          "WHERE Attendance = (SELECT MAX(Attendance) FROM matches)").show()

# f - Match with most number of goals
print("\nMatch with most number of goals")
spark.sql("SELECT Year, Datetime, City, `Home Team Name`, `Away Team Name`, "
          "(`Home Team Goals` + `Away Team Goals`) total_goals "
          "FROM matches "
          "ORDER BY total_goals DESC").show()

# g - Country with most final wins
print("\nCountry with most final wins")
resultsDf.groupBy(['Winner']).count().orderBy("count", ascending=False).show(1)

# h - All Participating countries in World cup
print("\nAll Participating countries in World cup")
spark.sql("SELECT `Home Team Name` country FROM matches "
          "UNION "
          "SELECT `Away Team Name` country FROM matches "
          "ORDER BY 1").show(100)

# i - Top 5 cities with most number of matches played
print("\nTop 5 cities with most number of matches played")
matchesDf.groupBy(['City']).count().orderBy("count", ascending=False).show(5)

# j - Top 5 interesting final matches
print("\nTop 5 interesting final matches")
spark.sql("SELECT Year, `Home Team Name` team1, `Home Team Goals` team1score,"
          " `Away Team Name` team2, `Away Team Goals` team2score, `Home Team Goals`+`Away Team Goals` total"
          " FROM matches WHERE Stage = 'Final'"
          "ORDER BY 6 DESC").show(5)

# 3 - Perform 5 queries using spark rdds

print("\nTask 3 - 5 queries using spark rdds")

# Convert df to rdd
matchesRdd = matchesDf.rdd
resultsRdd = resultsDf.rdd

# a - Count no. of world cups
print("\nNumber of world cups = " + str(resultsRdd.count()))

# b - Countries winning in world cup final
print("\nCountries winning in world cup final")
print(resultsRdd.map(lambda x: x[2]).distinct().collect())

# c - Countries played in world cup final but never won
win = resultsRdd.map(lambda x: x[2]).distinct().collect()
lose = resultsRdd.map(lambda x: x[3]).distinct().collect()
print("\nCountries played in world cup final but never won")
print(list(set(lose) - set(win)))

# d - Hosting cities
print("\nHosting cities")
print(matchesRdd.map(lambda x: (x[4], 1)).reduceByKey(add).collect())

# e - Number of times brazil won the world cup
print("\nNumber of times brazil won the world cup = " + str(resultsRdd.filter(lambda x: x[2] == 'Brazil').count()))























