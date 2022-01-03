yaml_Jobwzlk8toolkit = """apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {}
  namespace: ggr
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
---
apiVersion: batch/v1
kind: Job
metadata:
  name: ggr-yz-{}
  namespace: ggr
spec:
  template:
    metadata:
      labels:
        # Label is used as selector in the service.
        app: minio
    spec:
      volumes:
        - name: storage
          persistentVolumeClaim:
            claimName: {}
      containers:
      - name: wzlk8toolkit
        image: {}
        command: {}
        ports:
        - containerPort: 9000
        resources:
          requests:
            #memory: "32Gi"
            cpu: "8000m"
            memory: "32Gi"
          limits:
            cpu: "16000m"
            memory: "64Gi"
        volumeMounts:
        - name: storage               # must match the volume name, above
          mountPath: "/storage"
      restartPolicy: Never
  backoffLimit: 4
---
apiVersion: v1
kind: Service
metadata:
  name: minio-service-{}
  namespace: ggr
spec:
  type: NodePort
  ports:
    - port: 9000
      targetPort: 9000
      protocol: TCP
  selector:
    app: minio   
"""

yaml_minio_client = """
apiVersion: batch/v1
kind: Job
metadata:
  name: yz-minio-client
  namespace: ggr
spec:
  template:
    spec:
      containers:
      - name: minioclient
        image: minio/mc:latest
        command: ['sh', '-c', 'echo The app is running! && sleep 36000']
      restartPolicy: Never
  backoffLimit: 4"""
