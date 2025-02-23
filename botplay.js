const PokeClient = require('pokemon-showdown-api');
const client = new PokeClient();

client.connect();
client.on('ready', () => {
  client.login('username', 'password');
});

//Go through process to join private match and play

//Once in match we will use event listeners to play the game
