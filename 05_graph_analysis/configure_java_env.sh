#!/bin/bash

# Script to configure Java environment for GraphX and PySpark

echo "Configuring Java environment for GraphX and PySpark..."

# Check if Java is installed
if ! command -v java &> /dev/null; then
    echo "Java not found. Installing OpenJDK..."
    sudo apt-get update
    sudo apt-get install -y openjdk-11-jdk
else
    echo "Java is already installed."
fi

# Set JAVA_HOME environment variable
export JAVA_HOME=$(readlink -f /usr/bin/java | sed "s:/bin/java::")
echo "JAVA_HOME set to $JAVA_HOME"

# Add JAVA_HOME to .bashrc for persistence
echo "export JAVA_HOME=$JAVA_HOME" >> ~/.bashrc
echo "export PATH=\$PATH:\$JAVA_HOME/bin" >> ~/.bashrc

# Set up environment variables for PySpark
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3
echo "export PYSPARK_PYTHON=python3" >> ~/.bashrc
echo "export PYSPARK_DRIVER_PYTHON=python3" >> ~/.bashrc

# Create a configuration file for our project
mkdir -p /home/ubuntu/fake_news_detection/config
cat > /home/ubuntu/fake_news_detection/config/spark_config.sh << EOL
#!/bin/bash
export JAVA_HOME=$JAVA_HOME
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3
export SPARK_LOCAL_IP=127.0.0.1
EOL

chmod +x /home/ubuntu/fake_news_detection/config/spark_config.sh

echo "Java environment configuration completed."
echo "To use in new sessions, run: source /home/ubuntu/fake_news_detection/config/spark_config.sh"
