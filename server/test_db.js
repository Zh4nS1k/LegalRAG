const { MongoClient } = require('mongodb');
async function run() {
  const client = new MongoClient("mongodb://127.0.0.1:27017");
  await client.connect();
  const db = client.db('legally');
  const chat = await db.collection('chats').findOne({});
  console.log("Chat doc:", chat);
  const user = await db.collection('users').findOne({});
  console.log("User doc:", user);
  await client.close();
}
run();
