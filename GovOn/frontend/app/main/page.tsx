import Link from "next/link";

export default function MainPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen py-2">
      <main className="flex flex-col items-center justify-center w-full flex-1 px-20 text-center">
        <h1 className="text-4xl font-bold">Main Page</h1>
        <p className="mt-3 text-2xl">
          GovOn에 오신 것을 환영합니다.
        </p>
        <Link href="/" className="mt-8 rounded-lg bg-blue-600 px-5 py-2 text-white hover:bg-blue-700">
          홈으로 돌아가기
        </Link>
      </main>
    </div>
  );
}
