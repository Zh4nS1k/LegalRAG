const { MongoClient } = require('mongodb');
async function run() {
  const uri = "mongodb+srv://beathovenmozart:s1KDWX5%24@cluster0.zowoyqf.mongodb.net/legally?appName=Cluster0";
  const client = new MongoClient(uri);
  await client.connect();
  const db = client.db('legally_bot');
  const user = await db.collection('users').findOne({});
  console.log("User doc password field:", user.password, "Name:", user.name);
  await client.close();
}
run();
