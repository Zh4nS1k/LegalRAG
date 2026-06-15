const { MongoClient } = require('mongodb');
async function run() {
  const uri = "mongodb+srv://beathovenmozart:s1KDWX5%24@cluster0.zowoyqf.mongodb.net/legally?appName=Cluster0";
  const client = new MongoClient(uri);
  await client.connect();
  const db = client.db('legally_bot');
  const chat = await db.collection('chats').findOne({});
  console.log("Chat doc:", chat);
  const user = await db.collection('users').findOne({});
  console.log("User doc:", user);
  await client.close();
}
run();
