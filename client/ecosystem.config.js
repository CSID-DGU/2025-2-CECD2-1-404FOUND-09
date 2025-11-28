module.exports = {
  apps: [
    {
      name: 'fedhybrid-client',
      script: './.next/standalone/server.js',
      interpreter: 'node',
      cwd: '/home/jyh/fedhybrid-client',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      env: {
        NODE_ENV: 'production',
        PORT: 8081,
        HOST: '0.0.0.0'
      }
    }
  ]
}; 