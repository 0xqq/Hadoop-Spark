package com.friends;

import java.io.IOException;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class MutualFriends {

  public static class FriendMapper
          extends Mapper<Object, Text, Text, Text> {

      public void map(Object key, Text value, Context context
      ) throws IOException, InterruptedException {

          // Each line separated by newline character
          StringTokenizer itr = new StringTokenizer(value.toString(), "\n");
          String line = null;
          String[] lineSplit = null, friends = null;
          String f1, f2;
          int compare;
          Text keyFriends = new Text();

          while (itr.hasMoreTokens()) {
              // Get the first line
              line = itr.nextToken();

              // Split the line into user and friends by "->"
              // Input:  A -> B C D
              lineSplit = line.split(" -> ");

              // Get friends in array
              friends = lineSplit[1].split(" ");

              // Get the user as f1 (Need to sort)
              f1 = lineSplit[0];

              // Get friends in loop
              for (int i = 0; i < friends.length; i++) {

                  // Get friend in variable f2
                  f2 = friends[i];

                  // Sort
                  compare = f1.compareTo(f2);
                  if (compare < 0) {
                      keyFriends.set(f1 + " " + f2);
                  } else {
                      keyFriends.set(f2 + " " + f1);
                  }
                  context.write(keyFriends, new Text(": " + lineSplit[1]));
              }
          }
      }
  }


    public static class FriendReducer extends Reducer<Text, Text, Text, Text>{
        public void reduce(Text key, Iterator<Text> values,
                           OutputCollector<Text, Text> output, Reporter reporter) throws IOException{


            Text[] frndLists = new Text[2];

            int cnt = 0;
            while(values.hasNext()){
                frndLists[cnt++] = new Text(values.next());
            }

            //Get both list values into lists
            String[] list1 = frndLists[0].toString().split(" ");
            String[] list2 = frndLists[1].toString().split(" ");

            //Compare and get common values
            List<String> list = new LinkedList<String>();
            for(String friend1 : list1){
                for(String friend2 : list2){
                    if(friend1.equals(friend2)){
                        list.add(friend1);
                    }
                }
            }

            // Resulting common friends
            StringBuffer res = new StringBuffer();
            for(int i = 0; i < list.size(); i++){
                res.append(list.get(i));
                if(i != list.size() - 1)
                    res.append(" ");
            }
            output.collect(key, new Text(res.toString()));
        }
    }


  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "mutual friends");
    job.setJarByClass(MutualFriends.class);
    job.setMapperClass(FriendMapper.class);
    job.setCombinerClass(FriendReducer.class);
    job.setReducerClass(FriendReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}

